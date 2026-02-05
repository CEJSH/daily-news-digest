from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass
from typing import Callable


from daily_news_digest.processing.types import Item

@dataclass(frozen=True)
class _MatchPolicy:
    # substring 매칭을 허용하되, 너무 짧은 토큰(vocab)을 substring으로 쓰지 않도록 보호
    min_substring_vocab_len_en: int = 3   # e.g. "ai", "ev", "dc"는 substring 금지
    min_substring_vocab_len_ko: int = 2   # 한글은 1글자 substring 금지
    # clusterKey가 너무 빈약할 때(오탐 위험) merge를 막기 위한 최소 토큰 수
    min_cluster_tokens_for_merge: int = 2


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
        cluster_event_labels: dict[str, str],
        cluster_domains: dict[str, set[str]],
        cluster_relations: dict[str, set[str]],
        cluster_max_tokens: int,
        cluster_min_tokens_for_merge: int | None = None,
        cluster_max_entities: int,
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
        self._match_policy = _MatchPolicy(min_cluster_tokens_for_merge=int(cluster_min_tokens_for_merge or 2))
        for key in self._dedupe_event_groups.keys():
            self._dedupe_event_tokens.add(key.lower())
            self._dedupe_event_groups[key].add(key.lower())
        self._event_token_to_group: dict[str, str] = {}
        for group, vocab in self._dedupe_event_groups.items():
            for token in vocab:
                self._event_token_to_group[token] = group
        self._cluster_event_labels = cluster_event_labels or {}
        self._cluster_domains = {
            label: {t.lower() for t in tokens}
            for label, tokens in (cluster_domains or {}).items()
        }
        self._cluster_relations = {
            label: {t.lower() for t in tokens}
            for label, tokens in (cluster_relations or {}).items()
        }
        self._cluster_relation_labels = set(self._cluster_relations.keys())
        self._cluster_max_tokens = max(1, int(cluster_max_tokens or 1))
        self._cluster_max_entities = max(0, int(cluster_max_entities or 0))
        self._normalize_title_for_dedupe = normalize_title_for_dedupe_func
        self._normalize_token_for_dedupe = normalize_token_for_dedupe_func
        self._clean_text = clean_text_func
        self._jaccard = jaccard_func
        self._is_eligible = is_eligible_func or (lambda item: not item.get("dropReason"))
        self.seen_titles: set[str] = set()
        self.seen_title_tokens: list[tuple[set[str], Item]] = []
        self.seen_items_by_dedupe_key: dict[str, Item] = {}
        self.seen_dedupe_ngrams: list[tuple[set[str], Item]] = []
        self.seen_items_by_cluster_key: dict[str, Item] = {}

    def _rank_score(self, x: Item) -> float:
         # score 키가 없을 수 있으므로 항상 방어적으로 처리
        return float(x.get("aiImportance") or x.get("importance") or x.get("score") or 0.0)

    def tokenize_for_dedupe(self, text: str) -> list[str]:
        t = self._clean_text(text or "").lower()
        t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
        return [x for x in t.split() if x]


    def _safe_substring_vocab(self, vocab: set[str]) -> set[str]:
        # substring 매칭에 사용할 vocab은 너무 짧은 토큰을 제외(오탐 방지)
        safe: set[str] = set()
        for v in vocab:
            if not v:
                continue
            if self.is_korean_token(v):
                if len(v) >= self._match_policy.min_substring_vocab_len_ko:
                    safe.add(v)
            else:
                if len(v) >= self._match_policy.min_substring_vocab_len_en:
                    safe.add(v)
        return safe

    def _cluster_key_token_count(self, cluster_key: str) -> int:
        # "a/b/c" 형태 토큰 수
        return len([p for p in (cluster_key or "").split("/") if p])
    
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
            # 짧은 토큰은 dedupeKey에 들어오면 이후 cluster substring과 결합해 오탐을 유발
            if not self.valid_token_length(tok):
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
                # 보강 루프도 동일한 길이 규칙 적용(결정성/일관성)
                if not self.valid_token_length(tok):
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

    def _normalize_cluster_token(self, token: str) -> str:
        if not token:
            return ""
        t = self._clean_text(token).lower()
        t = re.sub(r"[^a-z0-9가-힣_]", "", t)
        return t

    def _detect_relation(self, tokens: list[str]) -> tuple[str, set[str]]:
        if not tokens:
            return "", set()
        tokens_set = set(tokens)
        for label in sorted(self._cluster_relation_labels, key=len, reverse=True):
            for tok in tokens:
                if label in tok:
                    related = {tok}
                    required = self._cluster_relations.get(label, set())
                    if required:
                        related |= (tokens_set & required)
                    return label, related
        for label, required in self._cluster_relations.items():
            if required and required.issubset(tokens_set):
                return label, set(required)
        return "", set()

    def _match_cluster_domains(self, tokens: list[str]) -> tuple[list[str], set[str]]:
        if not tokens:
            return [], set()
        tokens_set = set(tokens)
        matched_labels: list[str] = []
        matched_tokens: set[str] = set()
        # dict insertion order에 의존하지 않도록 label 정렬 (결정성 확보)
        for label in sorted(self._cluster_domains.keys()):
            vocab = self._cluster_domains[label]
            direct = tokens_set & vocab
            if direct:
                matched_labels.append(label)
                matched_tokens |= direct
                continue
            # substring 매칭은 "충분히 긴 vocab"만 허용(예: ai/ev/dc 같은 짧은 토큰 오탐 방지)
            safe_vocab = self._safe_substring_vocab(vocab)
            if not safe_vocab:
                continue
            for tok in tokens:
                if any(v in tok for v in safe_vocab):
                    matched_labels.append(label)
                    matched_tokens.add(tok)
                    break
        return matched_labels, matched_tokens
    

    def build_cluster_key(self, dedupe_key: str) -> str:
        tokens_raw = self._dedupe_key_tokens(dedupe_key)
        if not tokens_raw:
            return ""
        tokens: list[str] = []
        for tok in tokens_raw:
            norm = self._normalize_cluster_token(tok)
            if not norm:
                continue
            if self.is_noise_token(norm):
                continue
            tokens.append(norm)
        if not tokens:
            return ""

        relation_label, relation_tokens = self._detect_relation(tokens)
        event_groups = self._event_group_ids(set(tokens))
        event_labels = [
            self._normalize_cluster_token(self._cluster_event_labels.get(group, group))
            for group in sorted(event_groups)
        ]
        event_labels = [x for x in event_labels if x]

        domain_labels, domain_tokens = self._match_cluster_domains(tokens)
        domain_labels = [self._normalize_cluster_token(x) for x in domain_labels if x]

        event_token_set = set(self._dedupe_event_tokens)
        domain_token_set = set(domain_tokens)
        relation_token_set = set(relation_tokens)

        entity_tokens: list[str] = []
        for tok in tokens:
            if tok in event_token_set:
                continue
            if tok in domain_token_set:
                continue
            if tok in relation_token_set:
                continue
            if tok in entity_tokens:
                continue
            entity_tokens.append(tok)

        cluster_tokens: list[str] = []
        if relation_label and event_labels:
            cluster_tokens.append(f"{relation_label}_{event_labels[0]}")
            event_labels = event_labels[1:]
        elif relation_label:
            cluster_tokens.append(relation_label)

        for tok in event_labels:
            if tok not in cluster_tokens:
                cluster_tokens.append(tok)
        for tok in domain_labels:
            if tok not in cluster_tokens:
                cluster_tokens.append(tok)

        remaining_slots = max(0, self._cluster_max_tokens - len(cluster_tokens))
        entity_cap = remaining_slots if self._cluster_max_entities <= 0 else min(remaining_slots, self._cluster_max_entities)
        if entity_cap <= 0 and not cluster_tokens:
            entity_cap = self._cluster_max_tokens

        added_entities = 0
        for tok in entity_tokens:
            if len(cluster_tokens) >= self._cluster_max_tokens or added_entities >= entity_cap:
                break
            if tok not in cluster_tokens:
                cluster_tokens.append(tok)
                added_entities += 1

        if len(cluster_tokens) > self._cluster_max_tokens:
            cluster_tokens = cluster_tokens[: self._cluster_max_tokens]

        cluster_tokens = [t for t in cluster_tokens if t]
        return "/".join(cluster_tokens)

    def normalize_title_tokens(self, title: str) -> set[str]:
        return self._normalize_title_for_dedupe(title, self._stopwords)

    def find_existing_duplicate(
        self,
        tokens: set[str],
        *,
        seen_title_tokens: list[tuple[set[str], Item]] | None = None,
    ) -> Item | None:
        title_tokens = self.seen_title_tokens if seen_title_tokens is None else seen_title_tokens
        kept_item = next(
            (p_item for p_tok, p_item in title_tokens if self._jaccard(tokens, p_tok) >= self._title_dedupe_jaccard),
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
        cluster_key: str,
        item: Item,
    ) -> None:
        self.seen_titles.add(title_raw)
        self.seen_title_tokens.append((tokens, item))
        self.seen_items_by_dedupe_key[dedupe_key] = item
        if dedupe_ngrams:
            self.seen_dedupe_ngrams.append((dedupe_ngrams, item))
        if cluster_key:
            self.seen_items_by_cluster_key[cluster_key] = item

    def is_title_seen(self, title_raw: str) -> bool:
        return title_raw in self.seen_titles

    def _mark_merged(self, item: Item, matched: Item, reason: str) -> None:
        item["status"] = "merged"
        item["matchedTo"] = (
            matched.get("id")
            or matched.get("clusterKey")
            or matched.get("dedupeKey")
            or matched.get("title")
        )
        item["mergeReason"] = reason

    def apply_cluster_dedupe(self, items: list[Item]) -> None:
        candidates = sorted(items, key=self._rank_score, reverse=True)
        kept_by_cluster: dict[str, Item] = {}
        for item in candidates:
            if not self._is_eligible(item):
                continue
            key = (item.get("clusterKey") or "").strip()
            dedupe_key = item.get("dedupeKey") or ""
            if dedupe_key:
                recomputed = self.build_cluster_key(dedupe_key)
                if recomputed:
                    key = recomputed
                    item["clusterKey"] = recomputed
            if not key:
                continue
            matched = kept_by_cluster.get(key)
            if matched:
                # clusterKey가 너무 짧으면(오탐 위험) 바로 병합하지 않음
                if self._cluster_key_token_count(key) >= self._match_policy.min_cluster_tokens_for_merge:
                    self._mark_merged(item, matched, "cluster_duplicate")
                continue
            kept_by_cluster[key] = item

    def apply_dedupe_key_similarity(self, items: list[Item]) -> None:
        if self._dedupe_ngram_sim <= 0:
            return
        candidates = sorted(items, key=self._rank_score, reverse=True)
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
                        self._mark_merged(item, matched_key, "dedupe_key_exact")
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
                self._mark_merged(item, matched, "dedupe_key_sim")
                continue
            kept.append((ngrams, item))

    def apply_entity_event_dedupe(self, items: list[Item]) -> None:
        if not self._dedupe_event_tokens:
            return
        candidates = sorted(items, key=self._rank_score, reverse=True)
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
                self._mark_merged(item, matched, "entity_event_duplicate")
                continue
            kept.append((event_groups, entity_tokens, item))

    def apply_dedupe_key_prefix(self, items: list[Item], prefix_tokens: int = 3) -> None:
        if prefix_tokens <= 0:
            return
        candidates = sorted(
            items,
            key=lambda x: (
                x.get("aiImportance") or x.get("importance") or x.get("score", 0.0)
            ),
            reverse=True,
        )
        kept_by_prefix: dict[str, Item] = {}
        for item in candidates:
            if not self._is_eligible(item):
                continue
            key = (item.get("dedupeKey") or "").strip()
            if not key:
                continue
            parts = [p for p in key.split("-") if p]
            if len(parts) < prefix_tokens:
                continue
            prefix = "-".join(parts[:prefix_tokens])
            matched = kept_by_prefix.get(prefix)
            if matched:
                self._mark_merged(item, matched, "dedupe_key_prefix")
                continue
            kept_by_prefix[prefix] = item

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
                        cluster_key_raw = it.get("clusterKey")
                        key = cluster_key_raw or it.get("dedupeKey")
                        item_id = it.get("id")
                        if key and item_id:
                            if cluster_key_raw:
                                dedupe_map[cluster_key_raw] = item_id
                            else:
                                cluster_key = self.build_cluster_key(key) if key else ""
                                if cluster_key:
                                    dedupe_map[cluster_key] = item_id
                        if item_id:
                            title = it.get("title") or ""
                            summary = it.get("summary") or []
                            summary_text = (
                                " ".join(summary) if isinstance(summary, list) else str(summary)
                            )
                            alt_key = self.build_dedupe_key(title, summary_text)
                            alt_cluster = self.build_cluster_key(alt_key) if alt_key else ""
                            if alt_cluster:
                                dedupe_map[alt_cluster] = item_id
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
            cluster_key_raw = it.get("clusterKey")
            key = cluster_key_raw or it.get("dedupeKey")
            item_id = it.get("id")
            if key and item_id:
                if cluster_key_raw:
                    dedupe_map[cluster_key_raw] = item_id
                else:
                    cluster_key = self.build_cluster_key(key) if key else ""
                    if cluster_key:
                        dedupe_map[cluster_key] = item_id
            if item_id:
                title = it.get("title") or ""
                summary = it.get("summary") or []
                summary_text = " ".join(summary) if isinstance(summary, list) else str(summary)
                alt_key = self.build_dedupe_key(title, summary_text)
                alt_cluster = self.build_cluster_key(alt_key) if alt_key else ""
                if alt_cluster:
                    dedupe_map[alt_cluster] = item_id
        return dedupe_map
