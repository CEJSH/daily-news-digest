from __future__ import annotations

import math
import re
from typing import Any, Callable

from daily_news_digest.core.config import FULLTEXT_LOG_ENABLED, FULLTEXT_LOG_MAX_CHARS
from daily_news_digest.processing.constants import DEFAULT_TOP_MIX_TARGET
from daily_news_digest.processing.types import Item, LogFunc
from daily_news_digest.utils import clean_text

try:
    from daily_news_digest.scrapers.article_fetcher import FetchResult
except Exception:  # pragma: no cover - optional dependency
    FetchResult = None


class AIEnrichmentService:
    def __init__(
        self,
        *,
        enrich_item_with_ai_func: Callable[[Item], dict[str, Any]] | None,
        get_embedding_func: Callable[[str], list[float] | None] | None,
        fetch_article_text_func: Callable[..., Any] | None,
        estimate_read_time_func: Callable[[str], int],
        score_entry_func: Callable[[list[str], int, str | None], float],
        get_item_category_func: Callable[[Item], str] | None,
        is_eligible_func: Callable[[Item], bool],
        logger: LogFunc,
        ai_importance_enabled: bool,
        ai_importance_max_items: int,
        ai_importance_weight: float,
        ai_quality_enabled: bool,
        ai_semantic_dedupe_enabled: bool,
        ai_semantic_dedupe_max_items: int,
        ai_semantic_dedupe_threshold: float,
        article_fetch_enabled: bool,
        article_fetch_max_items: int,
        article_fetch_min_chars: int,
        article_fetch_timeout_sec: int,
        top_mix_target: dict[str, int] | None = None,
    ) -> None:
        self._enrich_item_with_ai = enrich_item_with_ai_func
        self._get_embedding = get_embedding_func
        self._fetch_article_text = fetch_article_text_func
        self._estimate_read_time_seconds = estimate_read_time_func
        self._score_entry = score_entry_func
        self._get_item_category = get_item_category_func
        self._is_eligible = is_eligible_func
        self._log = logger
        self._ai_importance_enabled = ai_importance_enabled
        self._ai_importance_max_items = ai_importance_max_items
        self._ai_importance_weight = ai_importance_weight
        self._ai_quality_enabled = ai_quality_enabled
        self._ai_semantic_dedupe_enabled = ai_semantic_dedupe_enabled
        self._ai_semantic_dedupe_max_items = ai_semantic_dedupe_max_items
        self._ai_semantic_dedupe_threshold = ai_semantic_dedupe_threshold
        self._article_fetch_enabled = article_fetch_enabled
        self._article_fetch_max_items = article_fetch_max_items
        self._article_fetch_min_chars = article_fetch_min_chars
        self._article_fetch_timeout_sec = article_fetch_timeout_sec
        self._top_mix_target = top_mix_target or DEFAULT_TOP_MIX_TARGET

    @staticmethod
    def _parse_meta_kv(meta_str: str) -> dict[str, str]:
        parts: dict[str, str] = {}
        for part in (meta_str or "").split("||"):
            if "=" in part:
                key, val = part.split("=", 1)
                parts[key] = val
        return parts

    def _normalize_fetch_result(
        self, res: Any
    ) -> tuple[str, str, int | None, str | None, list[str]]:
        text = ""
        resolved = ""
        status_code: int | None = None
        extractor: str | None = None
        notes_local: list[str] = []
        if FetchResult is not None and isinstance(res, FetchResult):
            text = res.text or ""
            meta = getattr(res, "meta", None)
            if meta:
                resolved = getattr(meta, "final_url", "") or ""
                status_raw = getattr(meta, "status", None)
                try:
                    status_code = int(status_raw) if status_raw is not None else None
                except Exception:
                    status_code = status_raw  # type: ignore[assignment]
                extractor = getattr(meta, "extractor", None) or extractor
                notes = getattr(meta, "notes", None)
                if isinstance(notes, list):
                    notes_local = [str(n) for n in notes if n is not None]
            return text, resolved, status_code, extractor, notes_local
        if isinstance(res, (tuple, list)):
            if len(res) > 0:
                text = res[0] or ""
            if len(res) > 1:
                resolved_meta = res[1] or ""
                if isinstance(resolved_meta, str) and "final_url=" in resolved_meta:
                    meta_parts = self._parse_meta_kv(resolved_meta)
                    resolved = meta_parts.get("final_url", "")
                    extractor = meta_parts.get("extractor", extractor)
                    meta_status = meta_parts.get("status")
                    try:
                        status_code = int(meta_status) if meta_status is not None else None
                    except Exception:
                        status_code = status_code
                else:
                    resolved = resolved_meta
            return text, resolved, status_code, extractor, notes_local
        text = getattr(res, "text", "") or ""
        resolved = getattr(res, "url", "") or getattr(res, "final_url", "") or resolved
        if hasattr(res, "status"):
            try:
                status_code = int(getattr(res, "status"))
            except Exception:
                status_code = None
        if hasattr(res, "extractor"):
            extractor = getattr(res, "extractor")
        return text, resolved, status_code, extractor, notes_local

    def _mark_uneditable(self, item: Item, reason: str) -> None:
        if item.get("dropReason"):
            return
        item["dropReason"] = f"fetch_failed:{reason}"
        item["status"] = "dropped"
        item["whyImportant"] = "본문 확보 실패로 판단 불가입니다."
        item["importanceRationale"] = "근거: 본문 확보 실패로 판단 불가입니다."
        item["aiQuality"] = "low_quality"
        item["quality_reason"] = reason
        item["aiQualityTags"] = ["fetch_failed"]

    def _log_full_text(self, item: Item, text: str, stage: str, notes: list[str] | None = None) -> None:
        if not FULLTEXT_LOG_ENABLED:
            return
        if text is None:
            text = ""
        snippet_src = text if text else "<empty>"
        snippet = re.sub(r"\s+", " ", snippet_src).strip()[:FULLTEXT_LOG_MAX_CHARS]
        title = (item.get("title") or "").replace("|", " ").strip()
        link = (item.get("resolvedUrl") or item.get("link") or "").strip()
        attempts_label = ""
        if notes:
            attempts: list[str] = []
            for note in notes:
                if isinstance(note, str) and note.startswith("fetch_url:"):
                    attempts.append(note.split("fetch_url:", 1)[1])
            if attempts:
                seen: set[str] = set()
                unique: list[str] = []
                for attempt in attempts:
                    if attempt in seen:
                        continue
                    seen.add(attempt)
                    unique.append(attempt)
                if len(unique) <= 3:
                    attempts_label = f" attempts={','.join(unique)}"
                else:
                    head = ",".join(unique[:3])
                    attempts_label = f" attempts={head}(+{len(unique) - 3})"
        self._log(
            f"FULLTEXT[{stage}] len={len(text)} title={title[:60]} url={link}{attempts_label} :: {snippet}"
        )

    def _pick_ai_importance_candidates(self, items: list[Item]) -> list[Item]:
        if not items:
            return []

        max_items = int(self._ai_importance_max_items or 0)
        if max_items <= 0:
            return []

        eligible = [item for item in items if self._is_eligible(item)]
        ranked = sorted(eligible, key=lambda x: x.get("score", 0.0), reverse=True)
        if not ranked:
            return []

        if self._get_item_category is None:
            return ranked[:max_items]

        buckets: dict[str, list[Item]] = {"IT": [], "경제": [], "글로벌": []}
        other: list[Item] = []
        for item in ranked:
            category = self._get_item_category(item)
            if category in buckets:
                buckets[category].append(item)
            else:
                other.append(item)

        base = {k: int(self._top_mix_target.get(k, 0) or 0) for k in buckets}
        base_total = sum(base.values())
        if base_total <= 0:
            return ranked[:max_items]

        extra = max(0, max_items - base_total)
        alloc = dict(base)
        if extra > 0:
            remainder = extra
            for key in alloc:
                add = int(extra * alloc[key] / base_total)
                alloc[key] += add
                remainder -= add
            if remainder > 0:
                order = sorted(base.keys(), key=lambda k: (-base[k], k))
                for i in range(remainder):
                    alloc[order[i % len(order)]] += 1

        picked: list[Item] = []
        seen: set[int] = set()

        def _append(item: Item) -> None:
            obj_id = id(item)
            if obj_id in seen:
                return
            seen.add(obj_id)
            picked.append(item)

        for key in buckets:
            for item in buckets[key][: alloc[key]]:
                _append(item)

        for item in other:
            if len(picked) >= max_items:
                break
            _append(item)

        for item in ranked:
            if len(picked) >= max_items:
                break
            _append(item)

        picked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return picked[:max_items]

    def apply_ai_importance(self, items: list[Item]) -> None:
        if not self._ai_importance_enabled:
            return
        if self._enrich_item_with_ai is None:
            return

        def _format_fetch_debug(item: Item, *, budget: int, need_fetch: bool) -> str:
            title = (item.get("title") or "").replace("|", " ").strip()
            title = title[:50]
            source = (item.get("source") or "").replace("|", " ").strip()
            score = item.get("score")
            age = item.get("ageHours")
            return f"title={title}|source={source}|score={score}|age={age}|budget={budget}|need_fetch={int(need_fetch)}"

        candidates = self._pick_ai_importance_candidates(items)
        self._log(f"AI 중요도 평가 시작: {len(candidates)}개")
        fetch_budget = self._article_fetch_max_items
        fetch_attempted = 0
        fetch_succeeded = 0
        fetch_dropped = 0
        fetch_errors: list[str] = []
        ai_enriched = 0
        ai_low_quality_dropped = 0
        total = len(candidates)
        for idx, item in enumerate(candidates, start=1):
            if idx == 1 or idx % 5 == 0 or idx == total:
                self._log(
                    f"AI 중요도+본문 확보 진행: {idx}/{total} "
                    f"(본문 fetch {fetch_succeeded}/{fetch_attempted}, 예산 {fetch_budget})"
                )
            full_text = item.get("fullText") or ""
            link = item.get("link") or ""
            need_fetch = len(full_text) < self._article_fetch_min_chars or "news.google.com" in link

            if self._article_fetch_enabled and self._fetch_article_text and fetch_budget > 0:
                if need_fetch:
                    fetch_attempted += 1
                    fetch_output = self._fetch_article_text(
                        link,
                        timeout_sec=self._article_fetch_timeout_sec,
                    )
                    text, resolved_url, status_code, extractor, notes_local = self._normalize_fetch_result(fetch_output)
                    fetch_budget -= 1
                    parsed_status = None
                    resolved_for_item = resolved_url
                    if resolved_url and "||STATUS:" in resolved_url:
                        resolved_for_item, parsed_status = resolved_url.split("||STATUS:", 1)
                    elif resolved_url and "final_url=" in resolved_url:
                        meta_parts = self._parse_meta_kv(resolved_url)
                        resolved_for_item = meta_parts.get("final_url", resolved_for_item)
                        parsed_status = meta_parts.get("status", parsed_status)
                    if resolved_for_item:
                        item["resolvedUrl"] = resolved_for_item

                    failure_reasons: list[str] = []
                    if text and len(text.strip()) >= 80:
                        item["fullText"] = text
                        fetch_succeeded += 1
                        self._log_full_text(item, text, "ai_importance", notes_local)
                    else:
                        failure_reasons.append("본문 80자 미만")

                    if extractor == "none":
                        failure_reasons.append("extractor=none")

                    if status_code is not None and status_code >= 400:
                        failure_reasons.append(f"http_error:{status_code}")

                    if parsed_status and parsed_status.isdigit():
                        if int(parsed_status) >= 400:
                            failure_reasons.append(f"http_error:{parsed_status}")

                    if any(str(n).startswith(("http_error", "request_error")) for n in notes_local):
                        failure_reasons.append("http_error")

                    if failure_reasons:
                        fetch_dropped += 1
                        reason_text = ";".join(failure_reasons)
                        self._mark_uneditable(item, reason_text)
                        self._log_full_text(item, text, "ai_importance_failed", notes_local)
                        base_err = resolved_for_item or resolved_url or link
                        status_label = parsed_status or (str(status_code) if status_code is not None else "")
                        notes_label = ",".join(notes_local) if notes_local else ""
                        info_parts = [p for p in [status_label, notes_label, reason_text] if p]
                        debug = _format_fetch_debug(item, budget=fetch_budget, need_fetch=need_fetch)
                        if info_parts:
                            fetch_errors.append(f"{'|'.join(info_parts)}:{base_err}|{debug}")
                        else:
                            fetch_errors.append(f"{base_err}|{debug}")
                        continue
                # need_fetch 끝

            # fetch를 시도하지 못한 경우에도 본문이 충분하면 통과
            full_text_len = len((item.get("fullText") or "").strip())
            if full_text_len < 80:
                reason = "본문 없음 혹은 80자 미만"
                if need_fetch and fetch_budget <= 0:
                    reason = "fetch_budget_exhausted"
                elif need_fetch and (not self._article_fetch_enabled or not self._fetch_article_text):
                    reason = "fetch_disabled"
                fetch_dropped += 1
                self._mark_uneditable(item, reason)
                base = item.get("link") or item.get("title", "")[:60]
                debug = _format_fetch_debug(item, budget=fetch_budget, need_fetch=need_fetch)
                fetch_errors.append(f"no_full_text|{reason}:{base}|{debug}")
                continue
            ai_result = self._enrich_item_with_ai(item)
            if not ai_result:
                continue
            ai_enriched += 1
            item["ai"] = ai_result
            ai_dedupe_key = ai_result.get("dedupe_key")
            if ai_dedupe_key:
                item["dedupeKey"] = ai_dedupe_key
            if self._ai_quality_enabled:
                quality_label = ai_result.get("quality_label")
                if quality_label:
                    item["aiQuality"] = quality_label
                if quality_label == "low_quality":
                    reason = ai_result.get("quality_reason") or "ai_low_quality"
                    item["dropReason"] = f"ai_low_quality:{reason}"
                    item["aiQualityTags"] = ai_result.get("quality_tags") or []
                    ai_low_quality_dropped += 1
                    continue
            ai_category = ai_result.get("category_label")
            if ai_category:
                item["aiCategory"] = ai_category
            impact_signals_ai = ai_result.get("impact_signals") or []
            if impact_signals_ai:
                merged = sorted(set((item.get("impactSignals") or []) + impact_signals_ai))
                item["impactSignals"] = merged
                read_time_sec = item.get("readTimeSec")
                if not read_time_sec:
                    summary_raw = item.get("summaryRaw") or item.get("summary") or ""
                    read_time_sec = self._estimate_read_time_seconds(summary_raw)
                    item["readTimeSec"] = read_time_sec
                item["score"] = self._score_entry(merged, read_time_sec, item.get("source"))
            importance = ai_result.get("importance_score")
            if not importance:
                continue
            item["aiImportance"] = importance
            item["score"] = max(0.0, item["score"] + (importance - 3) * self._ai_importance_weight)
        self._log(
            "AI 중요도 평가 완료 "
            f"(AI 적용 {ai_enriched}/{total}, low_quality 드롭 {ai_low_quality_dropped}, "
            f"본문 fetch 성공 {fetch_succeeded}/{fetch_attempted}, fetch 실패 드롭 {fetch_dropped}, 예산 잔여 {fetch_budget})"
        )
        if fetch_errors:
            sample = "; ".join(fetch_errors[:5])
            self._log(f"본문 fetch 실패/빈본문 샘플 {len(fetch_errors)}건: {sample}")

    def prefetch_full_text(self, items: list[Item]) -> None:
        if not self._article_fetch_enabled or not self._fetch_article_text:
            return
        candidates = [item for item in items if self._is_eligible(item)]
        if not candidates:
            return
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        fetch_budget = self._article_fetch_max_items
        fetch_attempted = 0
        fetch_succeeded = 0
        fetch_dropped = 0
        fetch_errors: list[str] = []

        def _format_fetch_debug(item: Item, *, budget: int, need_fetch: bool) -> str:
            title = (item.get("title") or "").replace("|", " ").strip()
            title = title[:50]
            source = (item.get("source") or "").replace("|", " ").strip()
            score = item.get("score")
            age = item.get("ageHours")
            return f"title={title}|source={source}|score={score}|age={age}|budget={budget}|need_fetch={int(need_fetch)}"

        self._log(f"본문 prefetch 시작: {len(candidates)}개 (예산 {fetch_budget})")
        for item in candidates:
            if fetch_budget <= 0:
                break
            full_text = item.get("fullText") or ""
            link = item.get("link") or ""
            need_fetch = len(full_text) < self._article_fetch_min_chars or "news.google.com" in link
            if not need_fetch:
                continue
            fetch_attempted += 1
            fetch_output = self._fetch_article_text(
                link,
                timeout_sec=self._article_fetch_timeout_sec,
            )
            text, resolved_url, status_code, extractor, notes_local = self._normalize_fetch_result(fetch_output)
            fetch_budget -= 1
            parsed_status = None
            resolved_for_item = resolved_url
            if resolved_url and "||STATUS:" in resolved_url:
                resolved_for_item, parsed_status = resolved_url.split("||STATUS:", 1)
            elif resolved_url and "final_url=" in resolved_url:
                meta_parts = self._parse_meta_kv(resolved_url)
                resolved_for_item = meta_parts.get("final_url", resolved_for_item)
                parsed_status = meta_parts.get("status", parsed_status)
            if resolved_for_item:
                item["resolvedUrl"] = resolved_for_item

            failure_reasons: list[str] = []
            if text and len(text.strip()) >= 80:
                item["fullText"] = text
                fetch_succeeded += 1
                self._log_full_text(item, text, "prefetch", notes_local)
            else:
                failure_reasons.append("본문 80자 미만")

            if extractor == "none":
                failure_reasons.append("extractor=none")

            if status_code is not None and status_code >= 400:
                failure_reasons.append(f"http_error:{status_code}")

            if parsed_status and parsed_status.isdigit():
                if int(parsed_status) >= 400:
                    failure_reasons.append(f"http_error:{parsed_status}")

            if any(str(n).startswith(("http_error", "request_error")) for n in notes_local):
                failure_reasons.append("http_error")

            if failure_reasons:
                fetch_dropped += 1
                reason_text = ";".join(failure_reasons)
                self._mark_uneditable(item, reason_text)
                self._log_full_text(item, text, "prefetch_failed", notes_local)
                base_err = resolved_for_item or resolved_url or link
                status_label = parsed_status or (str(status_code) if status_code is not None else "")
                notes_label = ",".join(notes_local) if notes_local else ""
                info_parts = [p for p in [status_label, notes_label, reason_text] if p]
                debug = _format_fetch_debug(item, budget=fetch_budget, need_fetch=need_fetch)
                if info_parts:
                    fetch_errors.append(f"{'|'.join(info_parts)}:{base_err}|{debug}")
                else:
                    fetch_errors.append(f"{base_err}|{debug}")

        self._log(
            "본문 prefetch 완료 "
            f"(성공 {fetch_succeeded}/{fetch_attempted}, 실패 드롭 {fetch_dropped}, 예산 잔여 {fetch_budget})"
        )
        if fetch_errors:
            sample = "; ".join(fetch_errors[:5])
            self._log(f"본문 prefetch 실패 샘플 {len(fetch_errors)}건: {sample}")

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _dedupe_text(self, item: Item) -> str:
        title = item.get("title") or ""
        full_text = item.get("fullText") or ""
        summary_raw = item.get("summaryRaw") or item.get("summary") or ""
        base = full_text if full_text else summary_raw
        return clean_text(f"{title} {base}")

    def apply_semantic_dedupe(self, items: list[Item]) -> None:
        if not self._ai_semantic_dedupe_enabled:
            return
        if self._get_embedding is None:
            return

        candidates = sorted(items, key=lambda x: x["score"], reverse=True)[: self._ai_semantic_dedupe_max_items]
        self._log(f"AI 중복 제거 시작: {len(candidates)}개")
        kept: list[Item] = []
        total = len(candidates)
        for idx, item in enumerate(candidates, start=1):
            if idx == 1 or idx % 10 == 0 or idx == total:
                self._log(f"AI 중복 제거 진행: {idx}/{total}")
            if not self._is_eligible(item):
                continue
            text = self._dedupe_text(item)
            if not text:
                continue
            embedding = self._get_embedding(text)
            if not embedding:
                continue
            item["embedding"] = embedding
            is_dup = False
            for ref in kept:
                ref_emb = ref.get("embedding")
                if not ref_emb:
                    continue
                sim = self._cosine_similarity(embedding, ref_emb)
                if sim >= self._ai_semantic_dedupe_threshold:
                    item["dropReason"] = f"semantic_duplicate:{ref.get('title','')[:60]}"
                    item["matchedTo"] = ref.get("id") or ref.get("dedupeKey") or ref.get("title")
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)
        self._log("AI 중복 제거 완료")

