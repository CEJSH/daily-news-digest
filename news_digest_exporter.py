import feedparser
import datetime
import os
import json
import re
import math
from utils import (
    clean_text, trim_title_noise, get_source_name,
    normalize_title_for_dedupe, jaccard, estimate_read_time_seconds
)
from config import (
    RSS_SOURCES,
    NEWSLETTER_TITLE,
    AFFILIATE_AD_TEXT,
    AFFILIATE_LINK,
    OUTPUT_JSON,
    SELECTION_CRITERIA,
    EDITOR_NOTE,
    QUESTION_OF_THE_DAY,
    TOP_LIMIT,
    MIN_SCORE,
    MAX_ENTRIES_PER_FEED,
    AI_IMPORTANCE_ENABLED,
    AI_IMPORTANCE_MAX_ITEMS,
    AI_IMPORTANCE_WEIGHT,
    AI_QUALITY_ENABLED,
    AI_SEMANTIC_DEDUPE_ENABLED,
    AI_SEMANTIC_DEDUPE_MAX_ITEMS,
    AI_SEMANTIC_DEDUPE_THRESHOLD,
    ARTICLE_FETCH_ENABLED,
    ARTICLE_FETCH_MAX_ITEMS,
    ARTICLE_FETCH_MIN_CHARS,
    ARTICLE_FETCH_TIMEOUT_SEC,
)
from constants import (
    HARD_EXCLUDE_KEYWORDS,
    HARD_EXCLUDE_URL_HINTS,
    EXCLUDE_KEYWORDS,
    SOURCE_TIER_A,
    SOURCE_TIER_B,
    STOPWORDS,
    IMPACT_SIGNALS_MAP,
    DEDUPE_NOISE_WORDS,
    EMOTIONAL_DROP_KEYWORDS,
    DROP_CATEGORIES,
    MONTH_TOKENS,
    LONG_IMPACT_SIGNALS,
    MEDIA_SUFFIXES,
)
try:
    from ai_enricher import enrich_item_with_ai, get_embedding
except Exception:  # pragma: no cover - optional dependency
    enrich_item_with_ai = None
    get_embedding = None
try:
    from article_fetcher import fetch_article_text
except Exception:  # pragma: no cover - optional dependency
    fetch_article_text = None
from export_manager import export_daily_digest_json
from html_generator import generate_html

# ==========================================
# 핵심 로직 함수
# ==========================================

def _log(message: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")

def get_impact_signals(text: str) -> list[str]:
    signals = []
    text_lower = text.lower()
    for signal, keywords in IMPACT_SIGNALS_MAP.items():
        if any(kw.lower() in text_lower for kw in keywords):
            signals.append(signal)
    return signals

def _tokenize_for_dedupe(text: str) -> list[str]:
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    return [x for x in t.split() if x]

def _is_korean_token(token: str) -> bool:
    return bool(re.search(r"[가-힣]", token))

def _is_noise_token(token: str) -> bool:
    if token in STOPWORDS or token in DEDUPE_NOISE_WORDS or token in MONTH_TOKENS:
        return True
    if token.isdigit():
        return True
    if re.search(r"\d", token):
        if token.endswith(("년", "월", "일")) and token[:-1].isdigit():
            return True
    if len(token) == 1:
        return True
    if any(token.endswith(suf) for suf in MEDIA_SUFFIXES):
        return True
    return False

def _valid_token_length(token: str) -> bool:
    if _is_korean_token(token):
        return len(token) >= 2
    return len(token) >= 3

def _strip_source_from_text(text: str, source_name: str) -> str:
    if not text or not source_name:
        return text
    src = re.escape(source_name.strip())
    cleaned = re.sub(rf"(?:\s*[\|\-–—·•:｜ㅣ]\s*)?{src}\s*\.{{0,3}}\s*$", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(rf"\s+{src}\s*\.{{0,3}}\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def get_dedupe_key(title: str, summary: str) -> str:
    # 1) 토큰화 및 노이즈 제거
    tokens = _tokenize_for_dedupe(f"{title} {summary}")

    # 2) 의미 있는 길이의 단어만 필터링 (4~8개 목표)
    seen = set()
    filtered: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        if _is_noise_token(tok) or not _valid_token_length(tok):
            continue
        filtered.append(tok)
        seen.add(tok)

    # 3) 부족할 경우 완화된 조건으로 보완
    if len(filtered) < 4:
        for tok in tokens:
            if tok in seen:
                continue
            if tok in STOPWORDS or tok in DEDUPE_NOISE_WORDS or tok in MONTH_TOKENS:
                continue
            if tok.isdigit() or len(tok) < 2:
                continue
            filtered.append(tok)
            seen.add(tok)
            if len(filtered) >= 4:
                break

    # 4) 8개 초과면 길이 우선으로 상위 8개 유지 (순서는 원래 등장 순서)
    if len(filtered) > 8:
        ranked = sorted(filtered, key=lambda x: (-len(x), filtered.index(x)))
        top = set(ranked[:8])
        filtered = [t for t in filtered if t in top][:8]

    if not filtered:
        fallback = [t for t in tokens if t][:4]
        filtered = fallback if fallback else ["news"]

    return "-".join(filtered).lower()

def map_topic_to_category(topic: str) -> str:
    t = (topic or "").lower()
    if t.startswith("it"): return "IT"
    if "경제" in t: return "경제"
    return "글로벌"

def _get_item_category(item: dict) -> str:
    return item.get("aiCategory") or map_topic_to_category(item.get("topic", ""))

def source_weight(source_name: str) -> float:
    s = (source_name or "").strip()
    if any(a in s for a in SOURCE_TIER_A): return 3.0
    if any(b in s for b in SOURCE_TIER_B): return 1.5
    return 0.3

def _compute_age_hours(entry) -> float | None:
    published_parsed = getattr(entry, "published_parsed", None)
    if not published_parsed:
        return None
    published_dt = datetime.datetime(*published_parsed[:6], tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = now - published_dt
    return delta.total_seconds() / 3600.0

def _passes_freshness(age_hours: float | None, impact_signals: list[str]) -> bool:
    if age_hours is None:
        return True
    if age_hours > 168:
        return False
    if age_hours > 72 and not any(s in LONG_IMPACT_SIGNALS for s in impact_signals):
        return False
    return True

def _passes_emotional_filter(category: str, text_all: str, impact_signals: list[str]) -> bool:
    if category in DROP_CATEGORIES:
        return False
    if any(k in text_all for k in EMOTIONAL_DROP_KEYWORDS):
        if any(s in LONG_IMPACT_SIGNALS for s in impact_signals):
            return True
        return False
    return True

def score_entry(impact_signals: list[str], read_time_sec: int) -> float:
    score = 0.0
    if any(s in LONG_IMPACT_SIGNALS for s in impact_signals):
        score += 3.0
    if any(s in ["capex", "infra", "security"] for s in impact_signals):
        score += 2.0
    if any(s in ["earnings", "market-demand"] for s in impact_signals):
        score += 1.0
    if read_time_sec <= 20:
        score += 0.5
    return score

def _is_eligible(item: dict) -> bool:
    return not item.get("dropReason")


def pick_top_with_mix(all_items, top_limit=5):
    buckets = {"IT": [], "경제": [], "글로벌": []}
    for it in all_items:
        if not _is_eligible(it):
            continue
        buckets[_get_item_category(it)].append(it)

    for cat in buckets:
        buckets[cat].sort(key=lambda x: x["score"], reverse=True)

    target = {"IT": 2, "경제": 2, "글로벌": 1}
    picked = []
    for cat, n in target.items():
        picked += buckets[cat][:n]

    if len(picked) < top_limit:
        remain = [
            x for x in sorted(all_items, key=lambda x: x["score"], reverse=True)
            if x not in picked and _is_eligible(x)
        ]
        picked += remain[: top_limit - len(picked)]

    return picked[:top_limit]

def _apply_ai_importance(items: list[dict]) -> None:
    if not AI_IMPORTANCE_ENABLED:
        return
    if enrich_item_with_ai is None:
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return

    candidates = sorted(items, key=lambda x: x["score"], reverse=True)[:AI_IMPORTANCE_MAX_ITEMS]
    _log(f"AI 중요도 평가 시작: {len(candidates)}개")
    fetch_budget = ARTICLE_FETCH_MAX_ITEMS
    total = len(candidates)
    for idx, item in enumerate(candidates, start=1):
        if idx == 1 or idx % 5 == 0 or idx == total:
            _log(f"AI 중요도 진행: {idx}/{total}")
        if ARTICLE_FETCH_ENABLED and fetch_article_text and fetch_budget > 0:
            full_text = item.get("fullText") or ""
            if len(full_text) < ARTICLE_FETCH_MIN_CHARS:
                text, resolved_url = fetch_article_text(
                    item.get("link") or "",
                    timeout_sec=ARTICLE_FETCH_TIMEOUT_SEC,
                )
                if text:
                    item["fullText"] = text
                if resolved_url:
                    item["resolvedUrl"] = resolved_url
                fetch_budget -= 1
        ai_result = enrich_item_with_ai(item)
        if not ai_result:
            continue
        item["ai"] = ai_result
        if AI_QUALITY_ENABLED:
            quality_label = ai_result.get("quality_label")
            if quality_label:
                item["aiQuality"] = quality_label
            if quality_label == "low_quality":
                reason = ai_result.get("quality_reason") or "ai_low_quality"
                item["dropReason"] = f"ai_low_quality:{reason}"
                item["aiQualityTags"] = ai_result.get("quality_tags") or []
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
                read_time_sec = estimate_read_time_seconds(summary_raw)
                item["readTimeSec"] = read_time_sec
            item["score"] = score_entry(merged, read_time_sec)
        importance = ai_result.get("importance_score")
        if not importance:
            continue
        item["aiImportance"] = importance
        item["score"] = max(0.0, item["score"] + (importance - 3) * AI_IMPORTANCE_WEIGHT)
    _log("AI 중요도 평가 완료")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dedupe_text(item: dict) -> str:
    title = item.get("title") or ""
    summary_raw = item.get("summaryRaw") or item.get("summary") or ""
    return clean_text(f"{title} {summary_raw}")


def _apply_semantic_dedupe(items: list[dict]) -> None:
    if not AI_SEMANTIC_DEDUPE_ENABLED:
        return
    if get_embedding is None:
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return

    candidates = sorted(items, key=lambda x: x["score"], reverse=True)[:AI_SEMANTIC_DEDUPE_MAX_ITEMS]
    _log(f"AI 중복 제거 시작: {len(candidates)}개")
    kept: list[dict] = []
    total = len(candidates)
    for idx, item in enumerate(candidates, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            _log(f"AI 중복 제거 진행: {idx}/{total}")
        if not _is_eligible(item):
            continue
        text = _dedupe_text(item)
        if not text:
            continue
        embedding = get_embedding(text)
        if not embedding:
            continue
        item["embedding"] = embedding
        is_dup = False
        for ref in kept:
            ref_emb = ref.get("embedding")
            if not ref_emb:
                continue
            sim = _cosine_similarity(embedding, ref_emb)
            if sim >= AI_SEMANTIC_DEDUPE_THRESHOLD:
                item["dropReason"] = f"semantic_duplicate:{ref.get('title','')[:60]}"
                item["matchedTo"] = ref.get("id") or ref.get("dedupeKey") or ref.get("title")
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
    _log("AI 중복 제거 완료")

def _load_yesterday_dedupe_map(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            digest = json.load(f)
    except Exception:
        return {}

    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    yesterday = (now_kst - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if digest.get("date") != yesterday:
        return {}

    items = digest.get("items", [])
    dedupe_map: dict[str, str] = {}
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
            alt_key = get_dedupe_key(title, summary_text)
            if alt_key:
                dedupe_map[alt_key] = item_id
    return dedupe_map

def fetch_news_grouped_and_top(sources, top_limit=3):
    _log("뉴스 수집 및 큐레이팅 시작")
    grouped_items, seen_titles, all_items, topic_limits = {}, set(), [], {}
    seen_title_tokens: list[tuple[set[str], dict]] = []
    seen_items_by_dedupe_key = {}
    yesterday_dedupe_map = _load_yesterday_dedupe_map(OUTPUT_JSON)

    for source_idx, source in enumerate(sources, start=1):
        topic, url, feed_limit = source["topic"], source["url"], source.get("limit", 3)
        topic_limits[topic] = max(topic_limits.get(topic, 0), feed_limit)
        _log(f"피드 로딩({source_idx}/{len(sources)}): {topic}")
        feed = feedparser.parse(url)
        _log(f"피드 항목 수: {len(feed.entries)}")

        total_entries = min(len(feed.entries), MAX_ENTRIES_PER_FEED)
        for entry_idx, entry in enumerate(feed.entries[:MAX_ENTRIES_PER_FEED], start=1):
            if entry_idx == 1 or entry_idx % 20 == 0 or entry_idx == total_entries:
                _log(f"항목 처리: {topic} {entry_idx}/{total_entries}")
            title = getattr(entry, "title", "").strip()
            summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            source_name = get_source_name(entry)
            summary_clean = clean_text(summary_raw)
            summary_clean = _strip_source_from_text(summary_clean, source_name)
            title_clean = trim_title_noise(clean_text(title), source_name)
            summary = (summary_clean[:200] + "...") if summary_clean else "내용을 확인하려면 클릭하세요."
            full_text = ""
            content_list = getattr(entry, "content", None)
            if isinstance(content_list, list) and content_list:
                parts = []
                for c in content_list:
                    value = ""
                    if isinstance(c, dict):
                        value = c.get("value", "") or ""
                    else:
                        value = getattr(c, "value", "") or ""
                    if value:
                        parts.append(value)
                if parts:
                    full_text = clean_text(" ".join(parts))
            if not full_text:
                full_text = summary_clean

            tokens = normalize_title_for_dedupe(title_clean, STOPWORDS)
            text_all = (title_clean + " " + summary_clean).lower()
            impact_signals = get_impact_signals(text_all)
            dedupe_key = get_dedupe_key(title_clean, summary_clean)
            matched_to = yesterday_dedupe_map.get(dedupe_key)

            kept_item = next((p_item for p_tok, p_item in seen_title_tokens if jaccard(tokens, p_tok) >= 0.6), None)
            if not kept_item:
                kept_item = seen_items_by_dedupe_key.get(dedupe_key)

            if kept_item:
                kept_item.setdefault("mergedSources", []).append({"title": title_clean, "link": entry.link, "source": get_source_name(entry)})
                continue

            if title in seen_titles: continue
            link = getattr(entry, "link", "") or ""
            category = map_topic_to_category(topic)
            age_hours = _compute_age_hours(entry)

            if any(bad.lower() in text_all for bad in HARD_EXCLUDE_KEYWORDS): continue
            if any(hint in link.lower() for hint in HARD_EXCLUDE_URL_HINTS): continue
            if any(bad.lower() in text_all for bad in EXCLUDE_KEYWORDS if bad not in EMOTIONAL_DROP_KEYWORDS): continue

            if matched_to:
                continue

            if not impact_signals:
                continue

            if not _passes_freshness(age_hours, impact_signals):
                continue

            if not _passes_emotional_filter(category, text_all, impact_signals):
                continue

            read_time_sec = estimate_read_time_seconds(summary_clean)
            score = score_entry(impact_signals, read_time_sec)
            if score < MIN_SCORE:
                continue

            seen_titles.add(title)
            item = {
                "title": title_clean, "link": entry.link, "summary": summary,
                "summaryRaw": summary_clean,
                "fullText": full_text,
                "published": getattr(entry, "published", None), "score": score,
                "topic": topic, "source": source_name,
                "impactSignals": impact_signals, "dedupeKey": dedupe_key, "matchedTo": matched_to,
                "readTimeSec": read_time_sec
            }
            seen_title_tokens.append((tokens, item))
            seen_items_by_dedupe_key[dedupe_key] = item
            grouped_items.setdefault(topic, []).append(item)
            all_items.append(item)
        _log(f"피드 완료: {topic}, 누적 수집 {len(all_items)}개")

    _apply_ai_importance(all_items)
    _apply_semantic_dedupe(all_items)

    for topic, items in grouped_items.items():
        filtered = [x for x in items if _is_eligible(x)]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        grouped_items[topic] = filtered[:topic_limits.get(topic, TOP_LIMIT)]

    return grouped_items, pick_top_with_mix(all_items, top_limit)

def main():
    try:
        _log("프로그램 시작")
        grouped_items, top_items = fetch_news_grouped_and_top(RSS_SOURCES, top_limit=TOP_LIMIT)
        
        config = {
            "newsletter_title": NEWSLETTER_TITLE,
            "ad_text": AFFILIATE_AD_TEXT,
            "ad_link": AFFILIATE_LINK,
            "selection_criteria": SELECTION_CRITERIA,
            "editor_note": EDITOR_NOTE,
            "question": QUESTION_OF_THE_DAY
        }

        export_daily_digest_json(top_items, OUTPUT_JSON, config)
        _log(f"완료! {OUTPUT_JSON} 파일이 생성되었습니다.")

    except Exception as e:
        print("❌ 오류 발생:", e)

if __name__ == "__main__":
    main()
