import datetime
import json
import math
import os
import re
from typing import Any, Callable

import feedparser

from config import (
    ARTICLE_FETCH_ENABLED,
    ARTICLE_FETCH_MAX_ITEMS,
    ARTICLE_FETCH_MIN_CHARS,
    ARTICLE_FETCH_TIMEOUT_SEC,
    FULLTEXT_LOG_ENABLED,
    FULLTEXT_LOG_MAX_CHARS,
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
    TOP_LIMIT,
    TOP_SOURCE_ALLOWLIST,
    TOP_SOURCE_ALLOWLIST_ENABLED,
    TOP_SOURCE_ALLOWLIST_STRICT,
    TOP_FRESH_MAX_HOURS,
    TOP_FRESH_EXCEPT_SIGNALS,
    TOP_FRESH_EXCEPT_MAX_HOURS,
    TOP_REQUIRE_PUBLISHED,
    SOURCE_WEIGHT_ENABLED,
    SOURCE_WEIGHT_FACTOR,
    TITLE_DEDUPE_JACCARD,
)
from constants import (
    DEDUPE_NOISE_WORDS,
    DEDUPE_EVENT_TOKENS,
    DEDUPE_EVENT_GROUPS,
    DROP_CATEGORIES,
    EMOTIONAL_DROP_KEYWORDS,
    EXCLUDE_KEYWORDS,
    HARD_EXCLUDE_KEYWORDS,
    HARD_EXCLUDE_URL_HINTS,
    LOCAL_PROMO_KEYWORDS,
    IMPACT_SIGNALS_MAP,
    LONG_IMPACT_SIGNALS,
    MEDIA_SUFFIXES,
    MONTH_TOKENS,
    SOURCE_TIER_A,
    SOURCE_TIER_B,
    STOPWORDS,
)
from utils import (
    clean_text,
    estimate_read_time_seconds,
    get_source_name,
    jaccard,
    normalize_title_for_dedupe,
    normalize_token_for_dedupe,
    trim_title_noise,
)

try:
    from ai_enricher import enrich_item_with_ai, get_embedding
except Exception:  # pragma: no cover - optional dependency
    enrich_item_with_ai = None
    get_embedding = None

try:
    from article_fetcher import fetch_article_text, FetchResult
except Exception:  # pragma: no cover - optional dependency
    fetch_article_text = None
    FetchResult = None

Item = dict[str, Any]
LogFunc = Callable[[str], None]
ParseFunc = Callable[[str], Any]

DEFAULT_TOP_MIX_TARGET = {"IT": 8, "경제": 8, "글로벌": 4}
SUMMARY_FALLBACK = "내용을 확인하려면 클릭하세요."


class EntryParser:
    def __init__(
        self,
        *,
        clean_text_func: Callable[[str], str],
        trim_title_noise_func: Callable[[str, str | None], str],
    ) -> None:
        self._clean_text = clean_text_func
        self._trim_title_noise = trim_title_noise_func

    def strip_source_from_text(self, text: str, source_name: str) -> str:
        if not text or not source_name:
            return text
        src = re.escape(source_name.strip())
        cleaned = re.sub(
            rf"(?:\s*[\|\-–—·•:｜ㅣ]\s*)?{src}\s*\.{{0,3}}\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(rf"\s+{src}\s*\.{{0,3}}\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def pick_analysis_text(self, full_text: str, summary_clean: str) -> str:
        if full_text:
            return full_text
        return summary_clean or ""

    def extract_full_text(self, entry: Any) -> str:
        content_list = getattr(entry, "content", None)
        if isinstance(content_list, list) and content_list:
            parts: list[str] = []
            for content in content_list:
                value = ""
                if isinstance(content, dict):
                    value = content.get("value", "") or ""
                else:
                    value = getattr(content, "value", "") or ""
                if value:
                    parts.append(value)
            if parts:
                return self._clean_text(" ".join(parts))
        return ""

    def parse_entry(self, entry: Any, source_name: str) -> tuple[str, str, str, str, str, str, str]:
        title_raw = getattr(entry, "title", "").strip()
        summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
        summary_clean = self._clean_text(summary_raw)
        summary_clean = self.strip_source_from_text(summary_clean, source_name)
        title_clean = self._trim_title_noise(self._clean_text(title_raw), source_name)
        summary = summary_clean if summary_clean else SUMMARY_FALLBACK
        full_text = self.extract_full_text(entry)
        analysis_text = self.pick_analysis_text(full_text, summary_clean)
        return (
            title_raw,
            title_clean,
            summary_raw,
            summary_clean,
            summary,
            full_text,
            analysis_text,
        )


class ItemFilterScorer:
    def __init__(
        self,
        *,
        impact_signals_map: dict[str, list[str]],
        long_impact_signals: set[str],
        emotional_drop_keywords: list[str],
        drop_categories: set[str],
        source_tier_a: set[str],
        source_tier_b: set[str],
        source_weight_enabled: bool,
        source_weight_factor: float,
        top_source_allowlist: set[str],
        top_source_allowlist_enabled: bool,
        top_fresh_max_hours: int,
        top_fresh_except_signals: set[str],
        top_fresh_except_max_hours: int,
        top_require_published: bool,
        now_provider: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._impact_signals_map = impact_signals_map
        self._long_impact_signals = long_impact_signals
        self._emotional_drop_keywords = emotional_drop_keywords
        self._drop_categories = drop_categories
        self._source_tier_a = source_tier_a
        self._source_tier_b = source_tier_b
        self._source_weight_enabled = source_weight_enabled
        self._source_weight_factor = source_weight_factor
        self._top_source_allowlist = {s for s in top_source_allowlist if s}
        self._top_source_allowlist_enabled = top_source_allowlist_enabled
        self._top_fresh_max_hours = top_fresh_max_hours
        self._top_fresh_except_signals = set(top_fresh_except_signals)
        self._top_fresh_except_max_hours = top_fresh_except_max_hours
        self._top_require_published = top_require_published
        self._now_provider = now_provider or (
            lambda: datetime.datetime.now(datetime.timezone.utc)
        )

    def get_impact_signals(self, text: str) -> list[str]:
        signals = []
        text_lower = text.lower()
        for signal, keywords in self._impact_signals_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                signals.append(signal)
        return signals

    def map_topic_to_category(self, topic: str) -> str:
        t = (topic or "").lower()
        if not t:
            return "글로벌"

        if "글로벌_빅테크" in t or "빅테크" in t:
            return "IT"
        if t.startswith("it") or "it" in t or "tech" in t:
            return "IT"
        if "ai" in t or "반도체" in t or "보안" in t or "저작권" in t or "데이터" in t:
            return "IT"

        if "글로벌_정세" in t or "정세" in t or "외교" in t:
            return "글로벌"

        if "국내" in t:
            return "경제"
        if "정책" in t or "규제" in t:
            return "경제"
        if "실적" in t or "가이던스" in t:
            return "경제"
        if "투자" in t or "ipo" in t or "m&a" in t or "ma" in t:
            return "경제"
        if "전력" in t or "인프라" in t or "에너지" in t:
            return "경제"
        if "경제" in t:
            return "경제"

        if "글로벌" in t or "global" in t:
            return "글로벌"
        return "글로벌"

    def get_item_category(self, item: Item) -> str:
        ai_category = item.get("aiCategory") or ""
        topic = item.get("topic", "")
        if ai_category == "글로벌" and "국내" in (topic or ""):
            return "경제"
        if ai_category in {"IT", "경제", "글로벌"}:
            return ai_category
        return self.map_topic_to_category(topic)

    def source_weight(self, source_name: str) -> float:
        s = (source_name or "").strip()
        if any(a in s for a in self._source_tier_a):
            return 3.0
        if any(b in s for b in self._source_tier_b):
            return 1.5
        return 0.3

    def source_weight_boost(self, source_name: str | None) -> float:
        if not self._source_weight_enabled:
            return 0.0
        if not source_name:
            return 0.0
        raw = self.source_weight(source_name)
        normalized = (raw - 0.3) / 2.7
        normalized = max(0.0, min(1.0, normalized))
        return normalized * self._source_weight_factor

    def compute_age_hours(self, entry: Any) -> float | None:
        published_parsed = getattr(entry, "published_parsed", None)
        if not published_parsed:
            return None
        published_dt = datetime.datetime(*published_parsed[:6], tzinfo=datetime.timezone.utc)
        now = self._now_provider()
        delta = now - published_dt
        return delta.total_seconds() / 3600.0

    def passes_freshness(self, age_hours: float | None, impact_signals: list[str]) -> bool:
        if age_hours is None:
            return True
        if age_hours > 168:
            return False
        if age_hours > 72 and not any(s in self._long_impact_signals for s in impact_signals):
            return False
        return True

    def passes_top_freshness(self, age_hours: float | None, impact_signals: list[str]) -> bool:
        if age_hours is None:
            return not self._top_require_published
        if age_hours <= self._top_fresh_max_hours:
            return True
        if any(s in self._top_fresh_except_signals for s in impact_signals):
            return age_hours <= self._top_fresh_except_max_hours
        return False

    def is_top_source_allowed(self, source_name: str | None) -> bool:
        if not self._top_source_allowlist_enabled:
            return True
        if not source_name:
            return False
        s = source_name.strip().lower()
        for allowed in self._top_source_allowlist:
            token = allowed.strip()
            if not token:
                continue
            if len(token) < 3:
                continue
            if token.lower() in s:
                return True
        return False

    def passes_emotional_filter(
        self,
        category: str,
        text_all: str,
        impact_signals: list[str],
    ) -> bool:
        if category in self._drop_categories:
            return False
        if any(k in text_all for k in self._emotional_drop_keywords):
            if any(s in self._long_impact_signals for s in impact_signals):
                return True
            return False
        return True

    def score_entry(
        self,
        impact_signals: list[str],
        read_time_sec: int,
        source_name: str | None = None,
    ) -> float:
        score = 0.0
        if any(s in self._long_impact_signals for s in impact_signals):
            score += 3.0
        if any(s in ["capex", "infra", "security"] for s in impact_signals):
            score += 2.0
        if any(s in ["earnings", "market-demand"] for s in impact_signals):
            score += 1.0
        if read_time_sec <= 20:
            score += 0.5
        score += self.source_weight_boost(source_name)
        return score

    def is_eligible(self, item: Item) -> bool:
        return not item.get("dropReason")

    def should_skip_entry(
        self,
        *,
        text_all: str,
        link_lower: str,
        matched_to: str | None,
        impact_signals: list[str],
        age_hours: float | None,
        category: str,
        hard_exclude_keywords: list[str],
        hard_exclude_url_hints: list[str],
        exclude_keywords: list[str],
        local_promo_keywords: list[str],
    ) -> bool:
        if any(bad in text_all for bad in hard_exclude_keywords):
            return True
        if any(hint in link_lower for hint in hard_exclude_url_hints):
            return True
        if any(bad in text_all for bad in local_promo_keywords):
            return True
        if any(bad in text_all for bad in exclude_keywords):
            return True
        if matched_to:
            return True
        if not impact_signals:
            return True
        if not self.passes_freshness(age_hours, impact_signals):
            return True
        if not self.passes_emotional_filter(category, text_all, impact_signals):
            return True
        return False


class DedupeEngine:
    def __init__(
        self,
        *,
        stopwords: set[str],
        dedupe_noise_words: set[str],
        month_tokens: set[str],
        media_suffixes: tuple[str, ...],
        title_dedupe_jaccard: float,
        dedupe_ngram_n: int,
        dedupe_ngram_sim: float,
        dedupe_event_tokens: set[str],
        dedupe_event_groups: dict[str, set[str]],
        normalize_title_for_dedupe_func: Callable[[str, set[str]], set[str]],
        normalize_token_for_dedupe_func: Callable[[str, set[str]], str],
        clean_text_func: Callable[[str], str],
        jaccard_func: Callable[[set[str], set[str]], float],
        is_eligible_func: Callable[[Item], bool] | None = None,
    ) -> None:
        self._stopwords = stopwords
        self._dedupe_noise_words = dedupe_noise_words
        self._month_tokens = month_tokens
        self._media_suffixes = media_suffixes
        self._title_dedupe_jaccard = title_dedupe_jaccard
        self._dedupe_ngram_n = dedupe_ngram_n
        self._dedupe_ngram_sim = dedupe_ngram_sim
        self._dedupe_event_tokens = {t.lower() for t in dedupe_event_tokens}
        self._dedupe_event_groups = {
            key: {t.lower() for t in tokens}
            for key, tokens in (dedupe_event_groups or {}).items()
        }
        for key in self._dedupe_event_groups.keys():
            self._dedupe_event_tokens.add(key.lower())
            self._dedupe_event_groups[key].add(key.lower())
        self._event_token_to_group: dict[str, str] = {}
        for group, vocab in self._dedupe_event_groups.items():
            for token in vocab:
                self._event_token_to_group[token] = group
        self._normalize_title_for_dedupe = normalize_title_for_dedupe_func
        self._normalize_token_for_dedupe = normalize_token_for_dedupe_func
        self._clean_text = clean_text_func
        self._jaccard = jaccard_func
        self._is_eligible = is_eligible_func or (lambda item: not item.get("dropReason"))
        self.seen_titles: set[str] = set()
        self.seen_title_tokens: list[tuple[set[str], Item]] = []
        self.seen_items_by_dedupe_key: dict[str, Item] = {}
        self.seen_dedupe_ngrams: list[tuple[set[str], Item]] = []

    def tokenize_for_dedupe(self, text: str) -> list[str]:
        t = self._clean_text(text or "").lower()
        t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
        return [x for x in t.split() if x]

    def is_korean_token(self, token: str) -> bool:
        return bool(re.search(r"[가-힣]", token))

    def is_noise_token(self, token: str) -> bool:
        if token in self._stopwords or token in self._dedupe_noise_words or token in self._month_tokens:
            return True
        if token.isdigit():
            return True
        if re.search(r"\d", token):
            if token.endswith(("년", "월", "일")) and token[:-1].isdigit():
                return True
        if len(token) == 1:
            return True
        if any(token.endswith(suf) for suf in self._media_suffixes):
            return True
        return False

    def valid_token_length(self, token: str) -> bool:
        if self.is_korean_token(token):
            return len(token) >= 2
        return len(token) >= 3

    def dedupe_key_ngrams(self, key: str, n: int | None = None) -> set[str]:
        if not key:
            return set()
        n_value = self._dedupe_ngram_n if n is None else n
        t = self._clean_text(key).lower()
        t = re.sub(r"[-\s]+", "", t)
        if len(t) < n_value:
            return set()
        return {t[i : i + n_value] for i in range(len(t) - n_value + 1)}

    def _dedupe_key_tokens(self, key: str) -> list[str]:
        if not key:
            return []
        t = self._clean_text(key).lower()
        return [p for p in t.split("-") if p]

    def _dedupe_core_tokens(self, key: str) -> list[str]:
        return [t for t in self._dedupe_key_tokens(key) if not re.search(r"\d", t)]

    def _event_group_ids(self, tokens: set[str]) -> set[str]:
        if not tokens:
            return set()
        groups: set[str] = set()
        for group, vocab in self._dedupe_event_groups.items():
            if tokens & vocab:
                groups.add(group)
        return groups

    def build_dedupe_key(self, title: str, summary: str) -> str:
        tokens = self.tokenize_for_dedupe(f"{title} {summary}")

        seen = set()
        filtered: list[str] = []
        for tok in tokens:
            tok = self._normalize_token_for_dedupe(tok, self._stopwords)
            if not tok:
                continue
            group = self._event_token_to_group.get(tok)
            if group:
                tok = group
            if tok in seen:
                continue
            if self.is_noise_token(tok):
                continue
            filtered.append(tok)
            seen.add(tok)

        if len(filtered) < 4:
            for tok in tokens:
                tok = self._normalize_token_for_dedupe(tok, self._stopwords)
                if not tok:
                    continue
                if tok in seen:
                    continue
                if tok in self._stopwords or tok in self._dedupe_noise_words or tok in self._month_tokens:
                    continue
                if tok.isdigit() or len(tok) < 2:
                    continue
                filtered.append(tok)
                seen.add(tok)
                if len(filtered) >= 4:
                    break

        if len(filtered) > 8:
            ranked = sorted(filtered, key=lambda x: (-len(x), filtered.index(x)))
            top = set(ranked[:8])
            filtered = [t for t in filtered if t in top][:8]

        if not filtered:
            fallback = [t for t in tokens if t][:4]
            filtered = fallback if fallback else ["news"]

        return "-".join(filtered).lower()

    def normalize_title_tokens(self, title: str) -> set[str]:
        return self._normalize_title_for_dedupe(title, self._stopwords)

    def find_existing_duplicate(
        self,
        tokens: set[str],
        dedupe_key: str,
        dedupe_ngrams: set[str],
        *,
        seen_title_tokens: list[tuple[set[str], Item]] | None = None,
        seen_items_by_dedupe_key: dict[str, Item] | None = None,
        seen_dedupe_ngrams: list[tuple[set[str], Item]] | None = None,
    ) -> Item | None:
        title_tokens = self.seen_title_tokens if seen_title_tokens is None else seen_title_tokens
        items_by_key = (
            self.seen_items_by_dedupe_key
            if seen_items_by_dedupe_key is None
            else seen_items_by_dedupe_key
        )
        dedupe_ngrams_list = (
            self.seen_dedupe_ngrams if seen_dedupe_ngrams is None else seen_dedupe_ngrams
        )
        kept_item = next(
            (p_item for p_tok, p_item in title_tokens if self._jaccard(tokens, p_tok) >= self._title_dedupe_jaccard),
            None,
        )
        if not kept_item:
            kept_item = items_by_key.get(dedupe_key)
        if not kept_item and dedupe_ngrams:
            kept_item = next(
                (
                    p_item
                    for p_ngrams, p_item in dedupe_ngrams_list
                    if self._jaccard(dedupe_ngrams, p_ngrams) >= self._dedupe_ngram_sim
                ),
                None,
            )
        return kept_item

    def record_item(
        self,
        *,
        title_raw: str,
        tokens: set[str],
        dedupe_key: str,
        dedupe_ngrams: set[str],
        item: Item,
    ) -> None:
        self.seen_titles.add(title_raw)
        self.seen_title_tokens.append((tokens, item))
        self.seen_items_by_dedupe_key[dedupe_key] = item
        if dedupe_ngrams:
            self.seen_dedupe_ngrams.append((dedupe_ngrams, item))

    def is_title_seen(self, title_raw: str) -> bool:
        return title_raw in self.seen_titles

    def apply_dedupe_key_similarity(self, items: list[Item]) -> None:
        if self._dedupe_ngram_sim <= 0:
            return
        candidates = sorted(items, key=lambda x: x["score"], reverse=True)
        seen_keys: dict[str, Item] = {}
        kept: list[tuple[set[str], Item]] = []
        for item in candidates:
            if not self._is_eligible(item):
                continue
            key = item.get("dedupeKey") or ""
            if key:
                key_norm = self._clean_text(key).lower()
                key_tokens = [p for p in key_norm.split("-") if p]
                if len(key_tokens) >= 2:
                    matched_key = seen_keys.get(key_norm)
                    if matched_key:
                        item["dropReason"] = f"dedupe_key_exact:{matched_key.get('title','')[:60]}"
                        item["matchedTo"] = (
                            matched_key.get("id")
                            or matched_key.get("dedupeKey")
                            or matched_key.get("title")
                        )
                        continue
                    seen_keys[key_norm] = item
            ngrams = self.dedupe_key_ngrams(key, self._dedupe_ngram_n)
            if not ngrams:
                continue
            matched = next(
                (
                    ref
                    for ref_ngrams, ref in kept
                    if self._jaccard(ngrams, ref_ngrams) >= self._dedupe_ngram_sim
                ),
                None,
            )
            if matched:
                item["dropReason"] = f"dedupe_key_sim:{matched.get('title','')[:60]}"
                item["matchedTo"] = matched.get("id") or matched.get("dedupeKey") or matched.get("title")
                continue
            kept.append((ngrams, item))

    def apply_entity_event_dedupe(self, items: list[Item]) -> None:
        if not self._dedupe_event_tokens:
            return
        candidates = sorted(items, key=lambda x: x["score"], reverse=True)
        kept: list[tuple[set[str], set[str], Item]] = []
        for item in candidates:
            if not self._is_eligible(item):
                continue
            key = item.get("dedupeKey") or ""
            tokens = set(self._dedupe_core_tokens(key))
            if not tokens:
                continue
            event_groups = self._event_group_ids(tokens)
            if not event_groups:
                continue
            entity_tokens = {t for t in tokens if t not in self._dedupe_event_tokens}
            if not entity_tokens:
                continue
            matched = next(
                (
                    ref
                    for ref_groups, ref_entities, ref in kept
                    if (event_groups & ref_groups) and (entity_tokens & ref_entities)
                ),
                None,
            )
            if matched:
                item["dropReason"] = f"entity_event_duplicate:{matched.get('title','')[:60]}"
                item["matchedTo"] = matched.get("id") or matched.get("dedupeKey") or matched.get("title")
                continue
            kept.append((event_groups, entity_tokens, item))

    def load_recent_dedupe_map(self, digest_path: str, history_path: str, days: int) -> dict[str, str]:
        if days <= 0:
            return {}
        now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        target_dates = {
            (now_kst - datetime.timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(1, days + 1)
        }
        dedupe_map: dict[str, str] = {}

        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = {}
            by_date = history.get("by_date") if isinstance(history, dict) else {}
            if isinstance(by_date, dict):
                for d in target_dates:
                    items = by_date.get(d) or []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        key = it.get("dedupeKey")
                        item_id = it.get("id")
                        if key and item_id:
                            dedupe_map[key] = item_id
                        if item_id:
                            title = it.get("title") or ""
                            summary_text = it.get("summary") or ""
                            alt_key = self.build_dedupe_key(title, summary_text)
                            if alt_key:
                                dedupe_map[alt_key] = item_id
                if dedupe_map:
                    return dedupe_map

        if not os.path.exists(digest_path):
            return dedupe_map
        try:
            with open(digest_path, "r", encoding="utf-8") as f:
                digest = json.load(f)
        except Exception:
            return dedupe_map
        if digest.get("date") not in target_dates:
            return dedupe_map
        items = digest.get("items", [])
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("status") not in {"published", "kept"}:
                continue
            key = it.get("dedupeKey")
            item_id = it.get("id")
            if key and item_id:
                dedupe_map[key] = item_id
            if item_id:
                title = it.get("title") or ""
                summary = it.get("summary") or []
                summary_text = " ".join(summary) if isinstance(summary, list) else str(summary)
                alt_key = self.build_dedupe_key(title, summary_text)
                if alt_key:
                    dedupe_map[alt_key] = item_id
        return dedupe_map


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
        item["whyImportant"] = f"요약 불가: {reason}"
        item["importanceRationale"] = f"근거: {reason}"
        item["aiQuality"] = "low_quality"
        item["quality_reason"] = reason
        item["aiQualityTags"] = ["fetch_failed"]

    def _log_full_text(self, item: Item, text: str, stage: str) -> None:
        if not FULLTEXT_LOG_ENABLED:
            return
        if text is None:
            text = ""
        snippet_src = text if text else "<empty>"
        snippet = re.sub(r"\s+", " ", snippet_src).strip()[:FULLTEXT_LOG_MAX_CHARS]
        title = (item.get("title") or "").replace("|", " ").strip()
        link = (item.get("link") or "").strip()
        self._log(f"FULLTEXT[{stage}] len={len(text)} title={title[:60]} url={link} :: {snippet}")

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

                    failure_reasons: list[str] = []
                    if text and len(text.strip()) >= 80:
                        item["fullText"] = text
                        fetch_succeeded += 1
                        self._log_full_text(item, text, "ai_importance")
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

                    if resolved_for_item:
                        item["resolvedUrl"] = resolved_for_item
                    if failure_reasons:
                        fetch_dropped += 1
                        reason_text = ";".join(failure_reasons)
                        self._mark_uneditable(item, reason_text)
                        self._log_full_text(item, text, "ai_importance_failed")
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

            failure_reasons: list[str] = []
            if text and len(text.strip()) >= 80:
                item["fullText"] = text
                fetch_succeeded += 1
                self._log_full_text(item, text, "prefetch")
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

            if resolved_for_item:
                item["resolvedUrl"] = resolved_for_item

            if failure_reasons:
                fetch_dropped += 1
                reason_text = ";".join(failure_reasons)
                self._mark_uneditable(item, reason_text)
                self._log_full_text(item, text, "prefetch_failed")
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

    def pick_top_with_mix(self, all_items: list[Item], top_limit: int = 5) -> list[Item]:
        def _pick_from_candidates(candidates: list[Item]) -> list[Item]:
            buckets: dict[str, list[Item]] = {"IT": [], "경제": [], "글로벌": []}
            for item in candidates:
                buckets[self._filter_scorer.get_item_category(item)].append(item)

            for category in buckets:
                buckets[category].sort(key=lambda x: x["score"], reverse=True)

            picked_local: list[Item] = []
            for category, limit in self._top_mix_target.items():
                picked_local += buckets[category][:limit]

            if len(picked_local) < top_limit:
                remain = [
                    x for x in sorted(candidates, key=lambda x: x["score"], reverse=True)
                    if x not in picked_local
                ]
                picked_local += remain[: top_limit - len(picked_local)]

            return picked_local[:top_limit]

        fresh_candidates = [
            item
            for item in all_items
            if self._filter_scorer.is_eligible(item)
            and self._filter_scorer.passes_top_freshness(
                item.get("ageHours"),
                item.get("impactSignals") or [],
            )
        ]
        allowlist_candidates = [
            item
            for item in fresh_candidates
            if self._filter_scorer.is_top_source_allowed(item.get("source"))
        ]

        if TOP_SOURCE_ALLOWLIST_ENABLED:
            if TOP_SOURCE_ALLOWLIST_STRICT:
                if len(allowlist_candidates) < top_limit:
                    self._log(f"⚠️ TOP allowlist 부족: {len(allowlist_candidates)}/{top_limit}")
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
        grouped_items: dict[str, list[Item]] = {}
        all_items: list[Item] = []
        total_seen = 0
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
                if entry_idx == total_entries:
                    self._log(
                        f"항목 처리: {topic} {entry_idx}/{total_entries} "
                        f"(누적 후보 {len(all_items)}개, 처리 {total_seen}개)"
                    )
                source_name = get_source_name(entry)
                (
                    title,
                    title_clean,
                    _summary_raw,
                    summary_clean,
                    summary,
                    full_text,
                    analysis_text,
                ) = self._entry_parser.parse_entry(entry, source_name)

                tokens = self._dedupe_engine.normalize_title_tokens(title_clean)
                text_all = (title_clean + " " + analysis_text).lower()
                impact_signals = self._filter_scorer.get_impact_signals(text_all)
                dedupe_key = self._dedupe_engine.build_dedupe_key(title_clean, analysis_text)
                dedupe_ngrams = self._dedupe_engine.dedupe_key_ngrams(dedupe_key, DEDUPKEY_NGRAM_N)
                matched_to = yesterday_dedupe_map.get(dedupe_key)

                kept_item = self._dedupe_engine.find_existing_duplicate(
                    tokens,
                    dedupe_key,
                    dedupe_ngrams,
                )
                if kept_item:
                    feed_dupes += 1
                    kept_item.setdefault("mergedSources", []).append(
                        {"title": title_clean, "link": entry.link, "source": source_name}
                    )
                    continue

                if self._dedupe_engine.is_title_seen(title):
                    feed_dupes += 1
                    continue
                link = getattr(entry, "link", "") or ""
                category = self._filter_scorer.map_topic_to_category(topic)
                age_hours = self._filter_scorer.compute_age_hours(entry)

                if self._filter_scorer.should_skip_entry(
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
                ):
                    feed_filtered += 1
                    continue

                read_time_sec = estimate_read_time_seconds(analysis_text)
                score = self._filter_scorer.score_entry(impact_signals, read_time_sec, source_name)
                if score < self._min_score:
                    feed_low_score += 1
                    continue

                item = {
                    "title": title_clean,
                    "link": entry.link,
                    "summary": summary,
                    "summaryRaw": summary_clean,
                    "fullText": full_text,
                    "published": getattr(entry, "published", None),
                    "score": score,
                    "topic": topic,
                    "source": source_name,
                    "impactSignals": impact_signals,
                    "dedupeKey": dedupe_key,
                    "matchedTo": matched_to,
                    "readTimeSec": read_time_sec,
                    "ageHours": age_hours,
                }
                self._dedupe_engine.record_item(
                    title_raw=title,
                    tokens=tokens,
                    dedupe_key=dedupe_key,
                    dedupe_ngrams=dedupe_ngrams,
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
        self._ai_service.apply_ai_importance(all_items)
        self._dedupe_engine.apply_entity_event_dedupe(all_items)
        self._ai_service.apply_semantic_dedupe(all_items)
        self._dedupe_engine.apply_dedupe_key_similarity(all_items)
        self._ai_service.prefetch_full_text(all_items)

        for topic, items in grouped_items.items():
            filtered = [x for x in items if self._filter_scorer.is_eligible(x)]
            filtered.sort(key=lambda x: x["score"], reverse=True)
            grouped_items[topic] = filtered[: topic_limits.get(topic, TOP_LIMIT)]

        top_items = self.pick_top_with_mix(all_items, top_limit)
        # 최종 선택 후보에 대해 본문을 최대한 확보 (편집 품질 보장)
        self._ai_service.prefetch_full_text(top_items)
        top_items = [item for item in top_items if self._filter_scorer.is_eligible(item)]
        if len(top_items) < top_limit:
            self._log(f"⚠️ 본문 확보 실패로 TOP 부족: {len(top_items)}/{top_limit}")
            refill = self.pick_top_with_mix(all_items, top_limit)
            self._ai_service.prefetch_full_text(refill)
            top_items = [item for item in refill if self._filter_scorer.is_eligible(item)]

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
