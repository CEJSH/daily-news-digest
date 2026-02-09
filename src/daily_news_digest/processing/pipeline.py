from __future__ import annotations

from typing import Any, Callable

import hashlib

import feedparser

from daily_news_digest.core.config import (
    ARTICLE_FETCH_ENABLED,
    ARTICLE_FETCH_MAX_ITEMS,
    ARTICLE_FETCH_MIN_CHARS,
    ARTICLE_FETCH_TIMEOUT_SEC,
    AI_IMPORTANCE_ENABLED,
    AI_IMPORTANCE_MAX_ITEMS,
    AI_IMPORTANCE_WEIGHT,
    AI_QUALITY_ENABLED,
    AI_SEMANTIC_DEDUPE_ENABLED,
    AI_SEMANTIC_DEDUPE_MAX_ITEMS,
    AI_SEMANTIC_DEDUPE_THRESHOLD,
    DEDUPE_HISTORY_PATH,
    DEDUPE_RECENT_DAYS,
    DEDUPKEY_NGRAM_N,
    DEDUPKEY_NGRAM_SIM,
    MAX_ENTRIES_PER_FEED,
    MIN_SCORE,
    OUTPUT_JSON,
    SIGNAL_CAP_ENABLED,
    SIGNAL_CAP_EXCEPT_IMPORTANCE,
    SIGNAL_CAP_EXCEPT_LONG_TRIGGER,
    SIGNAL_CAP_LABELS,
    SIGNAL_CAP_PENALTY,
    SIGNAL_CAP_RATIO,
    TOP_LIMIT,
    TOP_SOURCE_ALLOWLIST,
    TOP_SOURCE_ALLOWLIST_ENABLED,
    TOP_SOURCE_ALLOWLIST_STRICT,
    TOP_FRESH_EXCEPT_SIGNALS,
    TOP_FRESH_EXCEPT_MAX_HOURS,
    TOP_FRESH_MAX_HOURS,
    TOP_REQUIRE_PUBLISHED,
    SOURCE_WEIGHT_ENABLED,
    SOURCE_WEIGHT_FACTOR,
    TITLE_DEDUPE_JACCARD,
)
from daily_news_digest.core.constants import (
    DEDUPE_NOISE_WORDS,
    DEDUPE_EVENT_TOKENS,
    DEDUPE_EVENT_GROUPS,
    DEDUPE_CLUSTER_EVENT_LABELS,
    DEDUPE_CLUSTER_DOMAINS,
    DEDUPE_CLUSTER_RELATIONS,
    DEDUPE_CLUSTER_MAX_TOKENS,
    DEDUPE_CLUSTER_MAX_ENTITIES,
    DROP_CATEGORIES,
    POLICY_ACTION_KEYWORDS,
    POLITICAL_ACTOR_KEYWORDS,
    POLITICAL_COMMENTARY_KEYWORDS,
    EMOTIONAL_DROP_KEYWORDS,
    EXCLUDE_KEYWORDS,
    HARD_EXCLUDE_KEYWORDS,
    HARD_EXCLUDE_URL_HINTS,
    LOCAL_PROMO_KEYWORDS,
    IMPACT_SIGNALS_MAP,
    LONG_IMPACT_SIGNALS,
    MEDIA_SUFFIXES,
    MONTH_TOKENS,
    normalize_source_name,
    SOURCE_TIER_A,
    SOURCE_TIER_B,
    STOPWORDS,
)
from daily_news_digest.processing.ai_service import AIEnrichmentService
from daily_news_digest.processing.constants import (
    DEFAULT_TOP_MIX_MAX,
    DEFAULT_TOP_MIX_MIN,
    DEFAULT_TOP_MIX_TARGET,
    DEFAULT_TOP_SOURCE_MAX_PER_OUTLET,
)
from daily_news_digest.processing.dedupe import DedupeEngine
from daily_news_digest.processing.parsing import EntryParser
from daily_news_digest.processing.scoring import ItemFilterScorer
from daily_news_digest.processing.types import Item, LogFunc, ParseFunc
from daily_news_digest.utils import (
    clean_text,
    estimate_read_time_seconds,
    get_source_name,
    jaccard,
    normalize_title_for_dedupe,
    normalize_token_for_dedupe,
    parse_datetime_utc,
    trim_title_noise,
)

try:
    from daily_news_digest.processing.ai_enricher import enrich_item_with_ai, get_embedding
except Exception:  # pragma: no cover - optional dependency
    enrich_item_with_ai = None
    get_embedding = None

try:
    from daily_news_digest.scrapers.article_fetcher import fetch_article_text
except Exception:  # pragma: no cover - optional dependency
    fetch_article_text = None


class DigestPipeline:
    def __init__(
        self,
        *,
        entry_parser: EntryParser,
        filter_scorer: ItemFilterScorer,
        dedupe_engine: DedupeEngine,
        ai_service: AIEnrichmentService,
        feed_parser: ParseFunc,
        logger: LogFunc,
        max_entries_per_feed: int,
        min_score: float,
        output_json: str,
        dedupe_history_path: str,
        dedupe_recent_days: int,
        top_mix_target: dict[str, int] | None = None,
    ) -> None:
        self._entry_parser = entry_parser
        self._filter_scorer = filter_scorer
        self._dedupe_engine = dedupe_engine
        self._ai_service = ai_service
        self._feed_parser = feed_parser
        self._log = logger
        self._max_entries_per_feed = max_entries_per_feed
        self._min_score = min_score
        self._output_json = output_json
        self._dedupe_history_path = dedupe_history_path
        self._dedupe_recent_days = dedupe_recent_days
        self._top_mix_target = top_mix_target or DEFAULT_TOP_MIX_TARGET
        self._last_signal_cap_stats: dict[str, Any] = {}
        self._last_mix_stats: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}

    def get_signal_cap_stats(self) -> dict[str, Any]:
        return dict(self._last_signal_cap_stats)

    def get_mix_stats(self) -> dict[str, Any]:
        return dict(self._last_mix_stats)

    def get_pipeline_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def _log_allowlist_debug(self, candidates: list[Item], top_limit: int) -> None:
        if not candidates:
            self._log("allowlist 디버그: 후보 없음")
            return
        missing_source = 0
        unmatched: dict[str, int] = {}
        for item in candidates:
            source = item.get("source")
            source_raw = item.get("sourceRaw")
            if not source and not source_raw:
                missing_source += 1
            if self._filter_scorer.is_top_source_allowed(source, source_raw):
                continue
            label = (source_raw or source or "UNKNOWN").strip() or "UNKNOWN"
            unmatched[label] = unmatched.get(label, 0) + 1
        if missing_source:
            self._log(f"allowlist 미매칭: source 없음 {missing_source}개")
        if unmatched:
            top = sorted(unmatched.items(), key=lambda x: x[1], reverse=True)[:10]
            top_str = ", ".join([f"{name}:{count}" for name, count in top])
            self._log(f"allowlist 미매칭 소스 상위: {top_str}")
        self._log(
            "allowlist 디버그: "
            f"fresh_candidates={len(candidates)} "
            f"allowlist_size={len(TOP_SOURCE_ALLOWLIST)} "
            f"strict={int(TOP_SOURCE_ALLOWLIST_STRICT)} "
            f"top_limit={top_limit}"
        )

    def pick_top_with_mix(self, all_items: list[Item], top_limit: int = 5) -> list[Item]:
        from collections import defaultdict

        def _score(item: Item) -> float:
            try:
                return float(item.get("score") or 0.0)
            except Exception:
                return 0.0

        def _source_label(item: Item) -> str:
            return (item.get("source") or item.get("sourceRaw") or "UNKNOWN").strip() or "UNKNOWN"

        def _apply_signal_cap(
            picked_local: list[Item],
            candidates: list[Item],
            *,
            item_meta: dict[int, tuple[str, str]],
            counts_cat: dict[str, int],
            counts_src: dict[str, int],
            max_targets: dict[str, int],
            source_cap: int,
        ) -> list[Item]:
            cap_labels = {clean_text(str(s)).lower() for s in SIGNAL_CAP_LABELS if s}
            if not SIGNAL_CAP_ENABLED or not cap_labels:
                self._last_signal_cap_stats = {
                    "enabled": bool(SIGNAL_CAP_ENABLED),
                    "applied": False,
                    "replaced": 0,
                    "capLimit": 0,
                    "labels": sorted(cap_labels),
                    "ratio": SIGNAL_CAP_RATIO,
                    "penalty": SIGNAL_CAP_PENALTY,
                    "exceptLongTrigger": SIGNAL_CAP_EXCEPT_LONG_TRIGGER,
                    "exceptImportance": SIGNAL_CAP_EXCEPT_IMPORTANCE,
                }
                return picked_local
            cap_limit = max(1, int(top_limit * SIGNAL_CAP_RATIO))

            def _signals(item: Item) -> set[str]:
                raw = item.get("impactSignals") or []
                return {clean_text(str(s)).lower() for s in raw if s}

            def _is_capped(item: Item) -> bool:
                return bool(_signals(item) & cap_labels)

            def _is_exempt(item: Item) -> bool:
                importance = item.get("aiImportance") or item.get("importance") or 0
                try:
                    if int(importance) >= SIGNAL_CAP_EXCEPT_IMPORTANCE:
                        return True
                except Exception:
                    pass
                if not SIGNAL_CAP_EXCEPT_LONG_TRIGGER:
                    return False
                text_all = f"{item.get('title', '')} {item.get('summary', '')} {item.get('fullText', '')}"
                long_labels = self._filter_scorer.get_long_impact_labels(
                    text_all,
                    item.get("impactSignals") or [],
                )
                return bool(long_labels & cap_labels)

            capped_non_exempt = [p for p in picked_local if _is_capped(p) and not _is_exempt(p)]
            over = len(capped_non_exempt) - cap_limit
            if over <= 0:
                self._last_signal_cap_stats = {
                    "enabled": True,
                    "applied": False,
                    "replaced": 0,
                    "capLimit": cap_limit,
                    "labels": sorted(cap_labels),
                    "ratio": SIGNAL_CAP_RATIO,
                    "penalty": SIGNAL_CAP_PENALTY,
                    "exceptLongTrigger": SIGNAL_CAP_EXCEPT_LONG_TRIGGER,
                    "exceptImportance": SIGNAL_CAP_EXCEPT_IMPORTANCE,
                }
                return picked_local

            remaining = [c for c in candidates if c not in picked_local]
            remaining.sort(key=_score, reverse=True)
            removable = sorted(capped_non_exempt, key=_score)
            replaced = 0
            for victim in removable:
                if over <= 0:
                    break
                victim_meta = item_meta.get(id(victim))
                if not victim_meta:
                    continue
                v_cat, v_src = victim_meta
                for replacement in remaining:
                    if _is_capped(replacement) and not _is_exempt(replacement):
                        continue
                    r_meta = item_meta.get(id(replacement))
                    if not r_meta:
                        continue
                    r_cat, r_src = r_meta
                    counts_cat[v_cat] = max(0, counts_cat.get(v_cat, 0) - 1)
                    counts_src[v_src] = max(0, counts_src.get(v_src, 0) - 1)
                    can_add = True
                    if r_cat in max_targets and counts_cat.get(r_cat, 0) >= max_targets[r_cat]:
                        can_add = False
                    if source_cap > 0 and counts_src.get(r_src, 0) >= source_cap:
                        can_add = False
                    if can_add and _score(replacement) >= _score(victim) - SIGNAL_CAP_PENALTY:
                        picked_local.remove(victim)
                        picked_local.append(replacement)
                        remaining.remove(replacement)
                        counts_cat[r_cat] = counts_cat.get(r_cat, 0) + 1
                        counts_src[r_src] = counts_src.get(r_src, 0) + 1
                        over -= 1
                        replaced += 1
                        break
                    counts_cat[v_cat] = counts_cat.get(v_cat, 0) + 1
                    counts_src[v_src] = counts_src.get(v_src, 0) + 1

            if replaced > 0:
                self._log(
                    f"signal cap 적용: capped_limit={cap_limit} replaced={replaced}"
                )
            self._last_signal_cap_stats = {
                "enabled": True,
                "applied": replaced > 0,
                "replaced": replaced,
                "capLimit": cap_limit,
                "labels": sorted(cap_labels),
                "ratio": SIGNAL_CAP_RATIO,
                "penalty": SIGNAL_CAP_PENALTY,
                "exceptLongTrigger": SIGNAL_CAP_EXCEPT_LONG_TRIGGER,
                "exceptImportance": SIGNAL_CAP_EXCEPT_IMPORTANCE,
            }
            return picked_local

        def _pick_from_candidates(candidates: list[Item]) -> list[Item]:
            if not candidates:
                self._last_mix_stats = {
                    "topLimit": top_limit,
                    "minTargets": {},
                    "maxTargets": {},
                    "sourceCap": 0,
                    "picked": 0,
                    "categoryCounts": {},
                    "sourceCounts": {},
                    "minShortfall": {},
                    "minAdjusted": {},
                }
                return []

            item_meta: dict[int, tuple[str, str]] = {}
            buckets: dict[str, list[Item]] = {}
            for item in candidates:
                category = self._filter_scorer.get_item_category(item)
                source = _source_label(item)
                item_meta[id(item)] = (category, source)
                buckets.setdefault(category, []).append(item)
            for category in buckets:
                buckets[category].sort(key=_score, reverse=True)

            min_targets = dict(DEFAULT_TOP_MIX_MIN)
            max_targets = dict(DEFAULT_TOP_MIX_MAX)
            source_cap = int(DEFAULT_TOP_SOURCE_MAX_PER_OUTLET or 0)

            adjusted_min: dict[str, int] = {}
            for cat, val in min_targets.items():
                adjusted_min[cat] = min(int(val or 0), len(buckets.get(cat, [])))
            total_min = sum(adjusted_min.values())
            if total_min > top_limit:
                strengths = {
                    cat: (_score(buckets.get(cat, [])[0]) if buckets.get(cat) else 0.0)
                    for cat in adjusted_min
                }
                order = sorted(adjusted_min.keys(), key=lambda c: (strengths.get(c, 0.0), c))
                idx = 0
                while total_min > top_limit and order:
                    cat = order[idx % len(order)]
                    if adjusted_min[cat] > 0:
                        adjusted_min[cat] -= 1
                        total_min -= 1
                    idx += 1

            picked: list[Item] = []
            picked_ids: set[int] = set()
            counts_cat: dict[str, int] = defaultdict(int)
            counts_src: dict[str, int] = defaultdict(int)

            def _can_add(item: Item, *, enforce_max: bool = True, enforce_source: bool = True) -> bool:
                cat, src = item_meta.get(id(item), ("", ""))
                if enforce_max and cat in max_targets and counts_cat.get(cat, 0) >= max_targets[cat]:
                    return False
                if enforce_source and source_cap > 0 and counts_src.get(src, 0) >= source_cap:
                    return False
                return True

            for cat, need in adjusted_min.items():
                if need <= 0:
                    continue
                for item in buckets.get(cat, []):
                    if counts_cat.get(cat, 0) >= need:
                        break
                    if not _can_add(item, enforce_max=False):
                        continue
                    obj_id = id(item)
                    if obj_id in picked_ids:
                        continue
                    picked.append(item)
                    picked_ids.add(obj_id)
                    counts_cat[cat] += 1
                    counts_src[item_meta[obj_id][1]] += 1

            ranked = sorted(candidates, key=_score, reverse=True)
            for item in ranked:
                if len(picked) >= top_limit:
                    break
                obj_id = id(item)
                if obj_id in picked_ids:
                    continue
                if not _can_add(item):
                    continue
                picked.append(item)
                picked_ids.add(obj_id)
                cat, src = item_meta[obj_id]
                counts_cat[cat] += 1
                counts_src[src] += 1

            if len(picked) < top_limit and source_cap > 0:
                for item in ranked:
                    if len(picked) >= top_limit:
                        break
                    obj_id = id(item)
                    if obj_id in picked_ids:
                        continue
                    if not _can_add(item, enforce_source=False):
                        continue
                    picked.append(item)
                    picked_ids.add(obj_id)
                    cat, src = item_meta[obj_id]
                    counts_cat[cat] += 1
                    counts_src[src] += 1

            if len(picked) < top_limit:
                for item in ranked:
                    if len(picked) >= top_limit:
                        break
                    obj_id = id(item)
                    if obj_id in picked_ids:
                        continue
                    picked.append(item)
                    picked_ids.add(obj_id)
                    cat, src = item_meta[obj_id]
                    counts_cat[cat] += 1
                    counts_src[src] += 1

            picked = _apply_signal_cap(
                picked,
                candidates,
                item_meta=item_meta,
                counts_cat=counts_cat,
                counts_src=counts_src,
                max_targets=max_targets,
                source_cap=source_cap,
            )[:top_limit]

            min_shortfall = {
                cat: max(0, adjusted_min.get(cat, 0) - counts_cat.get(cat, 0))
                for cat in adjusted_min
                if adjusted_min.get(cat, 0) > counts_cat.get(cat, 0)
            }
            self._last_mix_stats = {
                "topLimit": top_limit,
                "minTargets": dict(min_targets),
                "minAdjusted": dict(adjusted_min),
                "maxTargets": dict(max_targets),
                "sourceCap": source_cap,
                "picked": len(picked),
                "categoryCounts": dict(counts_cat),
                "sourceCounts": dict(counts_src),
                "minShortfall": min_shortfall,
            }
            return picked[:top_limit]

        fresh_candidates = [
            item
            for item in all_items
            if self._filter_scorer.is_eligible(item)
            and self._filter_scorer.passes_top_freshness(
                item.get("ageHours"),
                item.get("impactSignals") or [],
                f"{item.get('title', '')} {item.get('summary', '')} {item.get('fullText', '')}",
            )
        ]
        allowlist_candidates = [
            item
            for item in fresh_candidates
            if self._filter_scorer.is_top_source_allowed(item.get("source"), item.get("sourceRaw"))
        ]

        if TOP_SOURCE_ALLOWLIST_ENABLED:
            if TOP_SOURCE_ALLOWLIST_STRICT:
                if len(allowlist_candidates) < top_limit:
                    self._log(f"⚠️ TOP allowlist 부족: {len(allowlist_candidates)}/{top_limit}")
                    self._log_allowlist_debug(fresh_candidates, top_limit)
                    picked = _pick_from_candidates(allowlist_candidates)
                    if len(picked) < top_limit:
                        remain = [
                            x for x in sorted(fresh_candidates, key=lambda x: x["score"], reverse=True)
                            if x not in picked
                        ]
                        picked += remain[: top_limit - len(picked)]
                    return picked[:top_limit]
                return _pick_from_candidates(allowlist_candidates)
            picked = _pick_from_candidates(allowlist_candidates)
            if len(picked) < top_limit:
                remain = [
                    x for x in sorted(fresh_candidates, key=lambda x: x["score"], reverse=True)
                    if x not in picked
                ]
                picked += remain[: top_limit - len(picked)]
            return picked[:top_limit]

        return _pick_from_candidates(fresh_candidates)

    def fetch_grouped_and_top(
        self,
        sources: list[dict[str, Any]],
        top_limit: int = 3,
    ) -> tuple[dict[str, list[Item]], list[Item]]:
        self._log("뉴스 수집 및 큐레이팅 시작")
        self._metrics = {
            "fetch": {
                "totalSeen": 0,
                "candidates": 0,
                "dupes": 0,
                "filtered": 0,
                "lowScore": 0,
                "skipReasons": {},
            },
            "dedupe": {},
            "selection": {},
        }
        grouped_items: dict[str, list[Item]] = {}
        all_items: list[Item] = []
        total_seen = 0
        item_seq = 0
        topic_limits: dict[str, int] = {}
        yesterday_dedupe_map = self._dedupe_engine.load_recent_dedupe_map(
            self._output_json,
            self._dedupe_history_path,
            self._dedupe_recent_days,
        )

        hard_exclude_keywords = [bad.lower() for bad in HARD_EXCLUDE_KEYWORDS]
        hard_exclude_url_hints = [hint.lower() for hint in HARD_EXCLUDE_URL_HINTS]
        exclude_keywords = [bad.lower() for bad in EXCLUDE_KEYWORDS if bad not in EMOTIONAL_DROP_KEYWORDS]
        local_promo_keywords = [bad.lower() for bad in LOCAL_PROMO_KEYWORDS]

        for source_idx, source in enumerate(sources, start=1):
            topic, url, feed_limit = source["topic"], source["url"], source.get("limit", 3)
            topic_limits[topic] = max(topic_limits.get(topic, 0), feed_limit)
            self._log(f"피드 로딩({source_idx}/{len(sources)}): {topic}")
            feed = self._feed_parser(url)
            self._log(f"피드 항목 수: {len(feed.entries)}")

            feed_seen = 0
            feed_kept = 0
            feed_dupes = 0
            feed_filtered = 0
            feed_low_score = 0
            total_entries = min(len(feed.entries), self._max_entries_per_feed)
            for entry_idx, entry in enumerate(feed.entries[: self._max_entries_per_feed], start=1):
                feed_seen += 1
                total_seen += 1
                self._metrics["fetch"]["totalSeen"] += 1
                if entry_idx == total_entries:
                    self._log(
                        f"항목 처리: {topic} {entry_idx}/{total_entries} "
                        f"(누적 후보 {len(all_items)}개, 처리 {total_seen}개)"
                    )
                source_name = get_source_name(entry)
                source_norm = normalize_source_name(source_name) or (source_name or "").strip()
                (
                    title,
                    title_clean,
                    summary,
                    full_text,
                    analysis_text,
                ) = self._entry_parser.parse_entry(entry, source_name)
                link = getattr(entry, "link", "") or ""
                published_raw = getattr(entry, "published", None)
                updated_raw = getattr(entry, "updated", None)
                published_at_utc = parse_datetime_utc(str(published_raw)) if published_raw else None
                updated_at_utc = parse_datetime_utc(str(updated_raw)) if updated_raw else None

                tokens = self._dedupe_engine.normalize_title_tokens(title_clean)
                text_all = (title_clean + " " + analysis_text).lower()
                impact_signals = list(self._filter_scorer.get_impact_signals(text_all))
                dedupe_input_text = f"{title_clean} {summary}".strip()
                dedupe_key = self._dedupe_engine.build_dedupe_key(title_clean, summary)
                cluster_hint = dedupe_input_text
                cluster_key = self._dedupe_engine.build_cluster_key(dedupe_key, hint_text=cluster_hint)
                input_hash = hashlib.sha1(clean_text(dedupe_input_text).encode("utf-8")).hexdigest()[:12]
                dedupe_ngrams = self._dedupe_engine.dedupe_key_ngrams(dedupe_key, DEDUPKEY_NGRAM_N)
                matched_to = yesterday_dedupe_map.get(cluster_key) if cluster_key else None

                kept_item = self._dedupe_engine.find_existing_duplicate(tokens)
                if kept_item:
                    feed_dupes += 1
                    self._metrics["fetch"]["dupes"] += 1
                    kept_item.setdefault("mergedSources", []).append(
                        {"title": title_clean, "link": link, "source": source_name}
                    )
                    continue

                if self._dedupe_engine.is_title_seen(title):
                    feed_dupes += 1
                    self._metrics["fetch"]["dupes"] += 1
                    continue
                category = self._filter_scorer.map_topic_to_category(topic)
                age_hours = self._filter_scorer.compute_age_hours(entry)

                skip_reason = self._filter_scorer.get_skip_reason(
                    text_all=text_all,
                    link_lower=link.lower(),
                    matched_to=matched_to,
                    impact_signals=impact_signals,
                    age_hours=age_hours,
                    category=category,
                    hard_exclude_keywords=hard_exclude_keywords,
                    hard_exclude_url_hints=hard_exclude_url_hints,
                    exclude_keywords=exclude_keywords,
                    local_promo_keywords=local_promo_keywords,
                )
                if skip_reason:
                    feed_filtered += 1
                    self._metrics["fetch"]["filtered"] += 1
                    reason_counts = self._metrics["fetch"]["skipReasons"]
                    reason_counts[skip_reason] = reason_counts.get(skip_reason, 0) + 1
                    continue

                read_time_sec = estimate_read_time_seconds(analysis_text)
                score = self._filter_scorer.score_entry(
                    impact_signals,
                    read_time_sec,
                    source_name,
                    text_all,
                )
                if score < self._min_score:
                    feed_low_score += 1
                    self._metrics["fetch"]["lowScore"] += 1
                    continue

                item_seq += 1
                item_id = f"item_{item_seq}"
                self._log(
                    f"item_build id={item_id} dedupe_input_hash={input_hash} "
                    f"dedupeKey={dedupe_key} clusterKey={cluster_key}"
                )
                item = {
                    "itemId": item_id,
                    "title": title_clean,
                    "link": link,
                    "summary": summary,
                    "fullText": full_text,
                    "published": published_raw,
                    "publishedAtUtc": published_at_utc.isoformat() if published_at_utc else "",
                    "updatedAtUtc": updated_at_utc.isoformat() if updated_at_utc else "",
                    "score": score,
                    "topic": topic,
                    "source": source_norm,
                    "sourceRaw": source_name,
                    "impactSignals": impact_signals,
                    "dedupeKey": dedupe_key,
                    "dedupeKeyRule": dedupe_key,
                    "clusterKey": cluster_key,
                    "clusterKeyRule": cluster_key,
                    "dedupeInputHash": input_hash,
                    "matchedTo": matched_to,
                    "readTimeSec": read_time_sec,
                    "ageHours": age_hours,
                }
                self._dedupe_engine.record_item(
                    title_raw=title,
                    tokens=tokens,
                    dedupe_key=dedupe_key,
                    dedupe_ngrams=dedupe_ngrams,
                    cluster_key=cluster_key,
                    item=item,
                )
                feed_kept += 1
                grouped_items.setdefault(topic, []).append(item)
                all_items.append(item)
            self._log(
                f"피드 완료: {topic} "
                f"(처리 {feed_seen}/{total_entries}, 후보 {feed_kept}개, 중복 {feed_dupes}개, "
                f"제외 {feed_filtered}개, 저점수 {feed_low_score}개, 누적 후보 {len(all_items)}개)"
            )

        self._log(f"수집 완료: 처리 {total_seen}개, 후보 {len(all_items)}개")
        self._metrics["fetch"]["candidates"] = len(all_items)
        self._ai_service.apply_ai_importance(all_items)
        before_cluster = sum(1 for x in all_items if x.get("mergeReason") == "cluster_duplicate")
        self._dedupe_engine.apply_cluster_dedupe(all_items)
        before_entity_event = sum(1 for x in all_items if x.get("mergeReason") == "entity_event_duplicate")
        self._dedupe_engine.apply_entity_event_dedupe(all_items)
        after_entity_event = sum(1 for x in all_items if x.get("mergeReason") == "entity_event_duplicate")
        before_key_sim = sum(1 for x in all_items if x.get("mergeReason") == "dedupe_key_sim")
        self._dedupe_engine.apply_dedupe_key_similarity(all_items)
        after_key_sim = sum(1 for x in all_items if x.get("mergeReason") == "dedupe_key_sim")
        after_cluster = sum(1 for x in all_items if x.get("mergeReason") == "cluster_duplicate")
        before_semantic = sum(1 for x in all_items if x.get("mergeReason") == "semantic_duplicate")
        self._ai_service.apply_semantic_dedupe(all_items)
        after_semantic = sum(1 for x in all_items if x.get("mergeReason") == "semantic_duplicate")
        self._metrics["dedupe"] = {
            "clusterMerged": max(0, after_cluster - before_cluster),
            "entityEventMerged": max(0, after_entity_event - before_entity_event),
            "dedupeKeySimMerged": max(0, after_key_sim - before_key_sim),
            "semanticMerged": max(0, after_semantic - before_semantic),
        }
        # 성능 최적화: 전체 후보에 대한 본문 prefetch는 비용이 크므로 생략

        for topic, items in grouped_items.items():
            filtered = [x for x in items if self._filter_scorer.is_eligible(x)]
            filtered.sort(key=lambda x: x["score"], reverse=True)
            grouped_items[topic] = filtered[: topic_limits.get(topic, TOP_LIMIT)]

        top_items = self.pick_top_with_mix(all_items, top_limit)
        self._metrics["selection"] = {
            "signalCap": self.get_signal_cap_stats(),
            "mix": self.get_mix_stats(),
        }
        # 최종 선택 후보에 대해 본문을 최대한 확보 (편집 품질 보장)
        self._ai_service.prefetch_full_text(top_items)
        top_items = [item for item in top_items if self._filter_scorer.is_eligible(item)]
        if len(top_items) < top_limit:
            self._log(f"⚠️ 본문 확보 실패로 TOP 부족: {len(top_items)}/{top_limit}")
            refill = self.pick_top_with_mix(all_items, top_limit)
            self._ai_service.prefetch_full_text(refill)
            top_items = [item for item in refill if self._filter_scorer.is_eligible(item)]

        if isinstance(self._metrics.get("selection"), dict):
            source_counts: dict[str, int] = {}
            category_counts: dict[str, int] = {}
            for item in top_items:
                source = (item.get("source") or item.get("sourceRaw") or "UNKNOWN").strip() or "UNKNOWN"
                source_counts[source] = source_counts.get(source, 0) + 1
                category = self._filter_scorer.get_item_category(item)
                category_counts[category] = category_counts.get(category, 0) + 1
            self._metrics["selection"].update(
                {
                    "finalCount": len(top_items),
                    "finalSources": source_counts,
                    "finalCategories": category_counts,
                }
            )

        return grouped_items, top_items


def build_default_entry_parser() -> EntryParser:
    return EntryParser(
        clean_text_func=clean_text,
        trim_title_noise_func=trim_title_noise,
    )


def build_default_filter_scorer() -> ItemFilterScorer:
    return ItemFilterScorer(
        impact_signals_map=IMPACT_SIGNALS_MAP,
        long_impact_signals=LONG_IMPACT_SIGNALS,
        emotional_drop_keywords=EMOTIONAL_DROP_KEYWORDS,
        drop_categories=DROP_CATEGORIES,
        political_actor_keywords=POLITICAL_ACTOR_KEYWORDS,
        political_commentary_keywords=POLITICAL_COMMENTARY_KEYWORDS,
        policy_action_keywords=POLICY_ACTION_KEYWORDS,
        source_tier_a=SOURCE_TIER_A,
        source_tier_b=SOURCE_TIER_B,
        source_weight_enabled=SOURCE_WEIGHT_ENABLED,
        source_weight_factor=SOURCE_WEIGHT_FACTOR,
        top_source_allowlist=TOP_SOURCE_ALLOWLIST,
        top_source_allowlist_enabled=TOP_SOURCE_ALLOWLIST_ENABLED,
        top_fresh_max_hours=TOP_FRESH_MAX_HOURS,
        top_fresh_except_signals=TOP_FRESH_EXCEPT_SIGNALS,
        top_fresh_except_max_hours=TOP_FRESH_EXCEPT_MAX_HOURS,
        top_require_published=TOP_REQUIRE_PUBLISHED,
    )


def build_default_dedupe_engine(
    *,
    is_eligible_func: Callable[[Item], bool] | None = None,
) -> DedupeEngine:
    return DedupeEngine(
        stopwords=STOPWORDS,
        dedupe_noise_words=DEDUPE_NOISE_WORDS,
        month_tokens=MONTH_TOKENS,
        media_suffixes=MEDIA_SUFFIXES,
        title_dedupe_jaccard=TITLE_DEDUPE_JACCARD,
        dedupe_ngram_n=DEDUPKEY_NGRAM_N,
        dedupe_ngram_sim=DEDUPKEY_NGRAM_SIM,
        dedupe_event_tokens=DEDUPE_EVENT_TOKENS,
        dedupe_event_groups=DEDUPE_EVENT_GROUPS,
        cluster_event_labels=DEDUPE_CLUSTER_EVENT_LABELS,
        cluster_domains=DEDUPE_CLUSTER_DOMAINS,
        cluster_relations=DEDUPE_CLUSTER_RELATIONS,
        cluster_max_tokens=DEDUPE_CLUSTER_MAX_TOKENS,
        cluster_max_entities=DEDUPE_CLUSTER_MAX_ENTITIES,
        normalize_title_for_dedupe_func=normalize_title_for_dedupe,
        normalize_token_for_dedupe_func=normalize_token_for_dedupe,
        clean_text_func=clean_text,
        jaccard_func=jaccard,
        is_eligible_func=is_eligible_func,
    )


def build_default_ai_service(
    *,
    logger: LogFunc,
    score_entry_func: Callable[[list[str], int, str | None], float],
    get_item_category_func: Callable[[Item], str] | None,
    is_eligible_func: Callable[[Item], bool],
) -> AIEnrichmentService:
    return AIEnrichmentService(
        enrich_item_with_ai_func=enrich_item_with_ai,
        get_embedding_func=get_embedding,
        fetch_article_text_func=fetch_article_text,
        estimate_read_time_func=estimate_read_time_seconds,
        score_entry_func=score_entry_func,
        get_item_category_func=get_item_category_func,
        is_eligible_func=is_eligible_func,
        logger=logger,
        ai_importance_enabled=AI_IMPORTANCE_ENABLED,
        ai_importance_max_items=AI_IMPORTANCE_MAX_ITEMS,
        ai_importance_weight=AI_IMPORTANCE_WEIGHT,
        ai_quality_enabled=AI_QUALITY_ENABLED,
        ai_semantic_dedupe_enabled=AI_SEMANTIC_DEDUPE_ENABLED,
        ai_semantic_dedupe_max_items=AI_SEMANTIC_DEDUPE_MAX_ITEMS,
        ai_semantic_dedupe_threshold=AI_SEMANTIC_DEDUPE_THRESHOLD,
        article_fetch_enabled=ARTICLE_FETCH_ENABLED,
        article_fetch_max_items=ARTICLE_FETCH_MAX_ITEMS,
        article_fetch_min_chars=ARTICLE_FETCH_MIN_CHARS,
        article_fetch_timeout_sec=ARTICLE_FETCH_TIMEOUT_SEC,
    )


def build_default_pipeline(*, logger: LogFunc) -> DigestPipeline:
    entry_parser = build_default_entry_parser()
    filter_scorer = build_default_filter_scorer()
    dedupe_engine = build_default_dedupe_engine(is_eligible_func=filter_scorer.is_eligible)
    ai_service = build_default_ai_service(
        logger=logger,
        score_entry_func=filter_scorer.score_entry,
        get_item_category_func=filter_scorer.get_item_category,
        is_eligible_func=filter_scorer.is_eligible,
    )
    return DigestPipeline(
        entry_parser=entry_parser,
        filter_scorer=filter_scorer,
        dedupe_engine=dedupe_engine,
        ai_service=ai_service,
        feed_parser=feedparser.parse,
        logger=logger,
        max_entries_per_feed=MAX_ENTRIES_PER_FEED,
        min_score=MIN_SCORE,
        output_json=OUTPUT_JSON,
        dedupe_history_path=DEDUPE_HISTORY_PATH,
        dedupe_recent_days=DEDUPE_RECENT_DAYS,
        top_mix_target=DEFAULT_TOP_MIX_TARGET,
    )
