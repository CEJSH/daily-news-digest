from __future__ import annotations

import datetime
import json
import os
import re
from typing import Callable

from daily_news_digest.processing.types import Item


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

