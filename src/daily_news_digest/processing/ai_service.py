from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

from daily_news_digest.core.config import FULLTEXT_LOG_ENABLED, FULLTEXT_LOG_MAX_CHARS
from daily_news_digest.processing.constants import DEFAULT_TOP_MIX_TARGET
from daily_news_digest.core.constants import (
    HARD_EXCLUDE_URL_HINTS,
    IMPACT_SIGNALS_MAP,
    MARKET_DEMAND_EVIDENCE_KEYWORDS,
    SANCTIONS_EVIDENCE_KEYWORDS,
    SECURITY_EVIDENCE_KEYWORDS,
    TRADE_TARIFF_KEYWORDS,
)
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
        self._hard_exclude_url_hints = [h.lower() for h in HARD_EXCLUDE_URL_HINTS]
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
        self._fetch_cache: dict[str, tuple[str, str, int | None, str | None, list[str]]] = {}

    def _should_exclude_url(self, url: str) -> bool:
        if not url:
            return False
        lowered = url.lower()
        return any(hint in lowered for hint in self._hard_exclude_url_hints)

    def _apply_url_hint_drop(self, item: Item, url: str) -> bool:
        if not self._should_exclude_url(url):
            return False
        item["dropReason"] = item.get("dropReason") or "hard_exclude_url_hint"
        item["status"] = "dropped"
        return True

    def _tokenize_for_overlap(self, text: str) -> set[str]:
        t = clean_text(text or "").lower()
        t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
        tokens: set[str] = set()
        for tok in t.split():
            if re.search(r"[가-힣]", tok):
                if len(tok) < 2:
                    continue
            else:
                if len(tok) < 3:
                    continue
            tokens.add(tok)
        return tokens

    def _log_integrity_drop(self, item: Item, reason: str) -> None:
        item_id = item.get("itemId") or ""
        link = item.get("link") or ""
        if not item_id:
            link_hash = hashlib.sha1(str(link).encode("utf-8")).hexdigest()[:10]
            item_id = f"link_{link_hash}"
        title = (item.get("title") or "")[:60]
        dedupe_key = (item.get("dedupeKey") or "")[:60]
        cluster_key = (item.get("clusterKey") or "")[:60]
        source = (item.get("source") or "")[:60]
        self._log(
            f"{reason} id={item_id} title={title} dedupeKey={dedupe_key} "
            f"clusterKey={cluster_key} source={source} link={link}"
        )

    def _integrity_precheck(self, items: list[Item]) -> None:
        for item in items:
            if not self._is_eligible(item):
                continue
            title = item.get("title") or ""
            dedupe_key = item.get("dedupeKey") or ""
            if title and dedupe_key:
                title_tokens = self._tokenize_for_overlap(title)
                dedupe_tokens = self._tokenize_for_overlap(dedupe_key.replace("-", " "))
                if title_tokens and dedupe_tokens:
                    overlap = len(title_tokens & dedupe_tokens) / max(1, len(title_tokens))
                    if overlap < 0.2:
                        item["status"] = "dropped"
                        item["dropReason"] = "ERROR: DEDUPE_KEY_TITLE_MISMATCH"
                        self._log_integrity_drop(item, "ERROR: DEDUPE_KEY_TITLE_MISMATCH")
                        continue

            ai_result = item.get("ai") if isinstance(item.get("ai"), dict) else {}
            if ai_result:
                title_ko = clean_text(ai_result.get("title_ko") or "")
                summary_lines = ai_result.get("summary_lines") or []
                if not isinstance(summary_lines, list):
                    summary_lines = []
                summary_ai = " ".join([str(x) for x in summary_lines if x])
                why_important = clean_text(ai_result.get("why_important") or "")
                ai_view_text = " ".join([title_ko, summary_ai, why_important]).strip()
                rule_summary = item.get("summary")
                if isinstance(rule_summary, list):
                    rule_summary_text = " ".join([str(x) for x in rule_summary if x])
                else:
                    rule_summary_text = str(rule_summary or "")
                rule_view_text = f"{title} {rule_summary_text}".strip()
                ai_tokens = self._tokenize_for_overlap(ai_view_text)
                rule_tokens = self._tokenize_for_overlap(rule_view_text)
                if ai_tokens and rule_tokens:
                    overlap = len(ai_tokens & rule_tokens) / max(1, len(rule_tokens))
                    if overlap < 0.2:
                        item["status"] = "dropped"
                        item["dropReason"] = "ERROR: MIXED_VIEW_FIELDS"
                        self._log_integrity_drop(item, "ERROR: MIXED_VIEW_FIELDS")

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

    def _fetch_with_cache(self, link: str) -> tuple[str, str, int | None, str | None, list[str]]:
        if not link:
            return "", "", None, None, []
        cached = self._fetch_cache.get(link)
        if cached is not None:
            return cached
        fetch_output = self._fetch_article_text(
            link,
            timeout_sec=self._article_fetch_timeout_sec,
        )
        result = self._normalize_fetch_result(fetch_output)
        self._fetch_cache[link] = result
        return result

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

        buckets: dict[str, list[Item]] = {k: [] for k in self._top_mix_target.keys()}
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
            item_id = item.get("itemId") or ""
            input_hash = item.get("dedupeInputHash") or ""
            title = (item.get("title") or "").replace("|", " ").strip()
            title = title[:50]
            source = (item.get("source") or "").replace("|", " ").strip()
            score = item.get("score")
            age = item.get("ageHours")
            return (
                f"id={item_id}|hash={input_hash}|title={title}|source={source}"
                f"|score={score}|age={age}|budget={budget}|need_fetch={int(need_fetch)}"
            )

        def _tokenize_basic(text: str) -> set[str]:
            t = clean_text(text or "").lower()
            t = re.sub(r"[^a-z0-9가-힣\s-]", " ", t)
            tokens: set[str] = set()
            for tok in re.split(r"[\s-]+", t):
                if not tok:
                    continue
                if re.search(r"[가-힣]", tok):
                    if len(tok) < 2:
                        continue
                else:
                    if len(tok) < 3:
                        continue
                tokens.add(tok)
            return tokens

        def _label_has_evidence(label: str, text_all: str) -> bool:
            if not label or not text_all:
                return False
            text = text_all.lower()
            if label == "sanctions":
                keywords = SANCTIONS_EVIDENCE_KEYWORDS
            elif label == "market-demand":
                keywords = MARKET_DEMAND_EVIDENCE_KEYWORDS
            elif label == "security":
                keywords = SECURITY_EVIDENCE_KEYWORDS
            elif label == "policy":
                keywords = IMPACT_SIGNALS_MAP.get("policy", []) + list(TRADE_TARIFF_KEYWORDS)
            else:
                keywords = IMPACT_SIGNALS_MAP.get(label, [])
            return any(kw.lower() in text for kw in keywords)

        def _format_snapshot(item: Item, stage: str) -> str:
            item_id = item.get("itemId") or ""
            title = (item.get("title") or "").replace("|", " ").strip()[:50]
            impact = [s for s in (item.get("impactSignals") or []) if isinstance(s, str)]
            dedupe_key = item.get("dedupeKey") or ""
            dedupe_rule = item.get("dedupeKeyRule") or ""
            cluster_key = item.get("clusterKey") or ""
            cluster_rule = item.get("clusterKeyRule") or ""
            ai_quality = item.get("aiQuality") or ""
            ai_importance = item.get("aiImportance") or ""
            return (
                f"AI_IMPORTANCE_{stage} id={item_id} title={title} "
                f"dedupeKey={dedupe_key} dedupeKeyRule={dedupe_rule} "
                f"clusterKey={cluster_key} clusterKeyRule={cluster_rule} "
                f"impactSignals={impact} aiQuality={ai_quality} aiImportance={ai_importance}"
            )

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
                    text, resolved_url, status_code, extractor, notes_local = self._fetch_with_cache(link)
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
                        if self._apply_url_hint_drop(item, resolved_for_item):
                            continue

                    failure_reasons: list[str] = []
                    if text and len(text.strip()) >= 50:
                        item["fullText"] = text
                        fetch_succeeded += 1
                        self._log_full_text(item, text, "ai_importance", notes_local)
                    else:
                        failure_reasons.append("본문 50자 미만")

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
            if full_text_len < 50:
                reason = "본문 없음 혹은 50자 미만"
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
            if self._apply_url_hint_drop(item, item.get("resolvedUrl") or link):
                continue
            summary_value = item.get("summary")
            if isinstance(summary_value, list):
                summary_text = " ".join([str(x) for x in summary_value if x])
            else:
                summary_text = str(summary_value or "")
            text_all = f"{item.get('title', '')} {item.get('summaryRaw', '')} {summary_text} {item.get('fullText', '')}"
            self._log(_format_snapshot(item, "PRE"))
            ai_result = self._enrich_item_with_ai(item)
            if not ai_result:
                self._log(_format_snapshot(item, "POST"))
                continue
            ai_enriched += 1
            item["ai"] = dict(ai_result)
            ai_dedupe_key = ai_result.get("dedupe_key")
            if ai_dedupe_key:
                item["dedupeKeyAI"] = ai_dedupe_key
                if item.get("dedupeKey"):
                    title_tokens = _tokenize_basic(item.get("title") or "")
                    ai_tokens = _tokenize_basic(str(ai_dedupe_key))
                    if len(title_tokens & ai_tokens) < 1:
                        item_id = item.get("itemId") or ""
                        self._log(
                            "WARN: DEDUPE_KEY_AI_NOT_ALIGNED "
                            f"id={item_id} ai_dedupe_key={ai_dedupe_key} title={(item.get('title') or '')[:50]}"
                        )
                else:
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
                    self._log(_format_snapshot(item, "POST"))
                    continue
            ai_category = ai_result.get("category_label")
            if ai_category:
                item["aiCategory"] = ai_category
            existing_signals = [
                s for s in (item.get("impactSignals") or []) if isinstance(s, str) and s
            ]
            item["impactSignals"] = existing_signals
            ai_labels_raw = ai_result.get("impact_signals") or []
            evidence_map = ai_result.get("impact_signals_evidence") or {}
            validated_ai: list[str] = []
            if isinstance(ai_labels_raw, list):
                for raw_label in ai_labels_raw:
                    if not isinstance(raw_label, str):
                        continue
                    label = clean_text(raw_label).lower()
                    if not label:
                        continue
                    evidence = clean_text(str(evidence_map.get(label) or ""))
                    if not evidence:
                        continue
                    if not _label_has_evidence(label, text_all):
                        continue
                    if label not in validated_ai:
                        validated_ai.append(label)
            if validated_ai:
                merged = existing_signals[:]
                for label in validated_ai:
                    if label not in merged:
                        merged.append(label)
                item["impactSignals"] = merged
                read_time_sec = item.get("readTimeSec")
                if not read_time_sec:
                    summary_raw = item.get("summaryRaw") or item.get("summary") or ""
                    read_time_sec = self._estimate_read_time_seconds(summary_raw)
                    item["readTimeSec"] = read_time_sec
                item["score"] = self._score_entry(merged, read_time_sec, item.get("source"), text_all)
            importance = ai_result.get("importance_score")
            if not importance:
                self._log(_format_snapshot(item, "POST"))
                continue
            item["aiImportance"] = importance
            item["score"] = max(0.0, item["score"] + (importance - 3) * self._ai_importance_weight)
            self._log(_format_snapshot(item, "POST"))
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
            item_id = item.get("itemId") or ""
            input_hash = item.get("dedupeInputHash") or ""
            title = (item.get("title") or "").replace("|", " ").strip()
            title = title[:50]
            source = (item.get("source") or "").replace("|", " ").strip()
            score = item.get("score")
            age = item.get("ageHours")
            return (
                f"id={item_id}|hash={input_hash}|title={title}|source={source}"
                f"|score={score}|age={age}|budget={budget}|need_fetch={int(need_fetch)}"
            )

        self._log(f"본문 prefetch 시작: {len(candidates)}개 (예산 {fetch_budget})")
        seen_links: set[str] = set()
        for item in candidates:
            if fetch_budget <= 0:
                break
            full_text = item.get("fullText") or ""
            link = item.get("link") or ""
            if len(full_text.strip()) >= self._article_fetch_min_chars:
                continue
            if link and link in seen_links:
                continue
            if link:
                seen_links.add(link)
            need_fetch = len(full_text) < self._article_fetch_min_chars or "news.google.com" in link
            if not need_fetch:
                continue
            fetch_attempted += 1
            text, resolved_url, status_code, extractor, notes_local = self._fetch_with_cache(link)
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
                if self._apply_url_hint_drop(item, resolved_for_item):
                    continue

            failure_reasons: list[str] = []
            if text and len(text.strip()) >= 50:
                item["fullText"] = text
                fetch_succeeded += 1
                self._log_full_text(item, text, "prefetch", notes_local)
            else:
                failure_reasons.append("본문 50자 미만")

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
        self._integrity_precheck(items)

        candidates = sorted(items, key=lambda x: x["score"], reverse=True)[: self._ai_semantic_dedupe_max_items]
        self._log(f"AI 중복 제거 시작: {len(candidates)}개")
        kept: list[Item] = []
        total = len(candidates)
        for idx, item in enumerate(candidates, start=1):
            if idx == 1 or idx % 10 == 0 or idx == total:
                self._log(f"AI 중복 제거 진행: {idx}/{total}")
            if not self._is_eligible(item):
                continue
            if item.get("status") == "merged":
                continue
            text = self._dedupe_text(item)
            if not text:
                continue
            embedding = self._get_embedding(text)
            if not embedding:
                continue
            item["embedding"] = embedding
            item_cluster = item.get("clusterKey") or ""
            is_dup = False
            for ref in kept:
                ref_emb = ref.get("embedding")
                if not ref_emb:
                    continue
                ref_cluster = ref.get("clusterKey") or ""
                if item_cluster and ref_cluster and item_cluster != ref_cluster:
                    continue
                sim = self._cosine_similarity(embedding, ref_emb)
                if sim >= self._ai_semantic_dedupe_threshold:
                    item["status"] = "merged"
                    item["matchedTo"] = ref.get("id") or ref.get("clusterKey") or ref.get("dedupeKey") or ref.get("title")
                    item["mergeReason"] = "semantic_duplicate"
                    is_dup = True
                    break
            if not is_dup:
                kept.append(item)
        self._log("AI 중복 제거 완료")
