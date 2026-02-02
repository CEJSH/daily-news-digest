import feedparser
import datetime
import os
import json
import re
from utils import (
    clean_text, trim_title_noise, get_source_name,
    normalize_title_for_dedupe, jaccard, estimate_read_time_seconds
)
from export_manager import export_daily_digest_json
from html_generator import generate_html

# ==========================================
# 사용자 설정 및 상수
# ==========================================

RSS_SOURCES = [
    {"topic": "IT", "url": "https://news.google.com/rss/search?q=AI+반도체+OR+데이터센터+OR+클라우드+OR+보안+취약점+OR+AI+규제+-리포트+-세미나+-웨비나+-칼럼&hl=ko&gl=KR&ceid=KR:ko", "limit": 15},
    {"topic": "IT", "url": "https://news.google.com/rss/search?q=AI+chips+OR+data+center+OR+cloud+infrastructure+OR+cybersecurity+vulnerability+OR+AI+regulation+-opinion+-column+-webinar+-whitepaper&hl=en&gl=US&ceid=US:en", "limit": 15},
    {"topic": "경제", "url": "https://news.google.com/rss/search?q=금리+OR+환율+OR+물가+OR+고용+OR+실적+OR+경기+전망+OR+정부+정책+OR+에너지전환+OR+태양광+OR+바이오+헬스케어+-리포트+-세미나+-칼럼&hl=ko&gl=KR&ceid=KR:ko", "limit": 15},
    {"topic": "경제", "url": "https://news.google.com/rss/search?q=interest+rate+OR+inflation+OR+fx+OR+jobs+report+OR+earnings+OR+economic+policy+OR+energy+transition+OR+biotech+OR+healthcare+-opinion+-column+-webinar+-whitepaper&hl=en&gl=US&ceid=US:en", "limit": 15},
    {"topic": "글로벌_정세", "url": "https://news.google.com/rss/search?q=관세+OR+제재+OR+무역+OR+공급망+OR+외교+OR+국제+협상+-사망+-살인+-폭행+-연예+-스포츠+-리포트+-칼럼&hl=ko&gl=KR&ceid=KR:ko", "limit": 15},
    {"topic": "글로벌_정세", "url": "https://news.google.com/rss/search?q=tariff+OR+sanctions+OR+trade+OR+supply+chain+OR+diplomacy+OR+geopolitics+-opinion+-column+-sports+-celebrity+-webinar+-whitepaper&hl=en&gl=US&ceid=US:en", "limit": 15},
    {"topic": "글로벌_빅테크", "url": "https://news.google.com/rss/search?q=Apple+OR+Microsoft+OR+Google+OR+OpenAI+OR+NVIDIA+OR+Amazon+OR+Meta+OR+Tesla+OR+TSMC+-opinion+-column+-webinar+-whitepaper&hl=en&gl=US&ceid=US:en", "limit": 15},
    {"topic": "글로벌_빅테크", "url": "https://news.google.com/rss/search?q=애플+OR+마이크로소프트+OR+구글+OR+오픈AI+OR+엔비디아+OR+아마존+OR+메타+OR+TSMC+-리포트+-세미나+-칼럼&hl=ko&gl=KR&ceid=KR:ko", "limit": 10},
]

QUALITY_KEYWORDS = ["분석", "해설", "전망", "심층", "진단", "전략", "패권", "패러다임", "변곡점", "구조", "재편", "지형", "모멘텀", "구조적", "생태계", "시나리오", "data", "in-depth", "diagnosis", "strategy", "paradigm", "inflection point", "structure", "reorganization", "ecosystem", "scenario"]
HARD_EXCLUDE_KEYWORDS = ["동향", "동향리포트", "리포트", "브리프", "백서", "자료집", "보고서", "연구보고서", "세미나", "웨비나", "컨퍼런스", "포럼", "행사", "모집", "신청", "접수", "보도자료", "홍보", "프로모션", "할인", "출시기념", "사설","칼럼","기고","기자수첩", "whitepaper", "report", "brief", "webinar", "conference", "forum", "press release", "promotion", "apply now", "opinion","editorial","column","commentary","view","must","should"]
HARD_EXCLUDE_URL_HINTS = ["/report", "/whitepaper", "/webinar", "/seminar", "/conference", "/event", "/download"]
EXCLUDE_KEYWORDS = ["연예", "스타", "걸그룹", "보이그룹", "아이돌", "배우", "가수", "예능", "드라마", "영화", "팬미팅", "컴백", "앨범", "뮤직비디오", "뮤비", "티저", "화보", "열애", "결별", "이혼", "결혼", "출산", "야구", "축구", "농구", "배구", "골프", "e스포츠", "K리그", "KBO", "프리미어리그", "챔피언스리그", "살해", "살인", "폭행", "성폭행", "강간", "납치", "사망", "시신",  "징역", "맛집", "카페", "뷰맛집", "여행기", "관광지", "연휴", "날씨", "미세먼지", "교통통제", "경악", "발칵", "알고보니", "이유는", "근황", "포착", "망신", "누리꾼", "갑론을박", "결국", "정체", "충격", "헉", "소름", "이게 얼마", "대참사", "대박", "주의보", "레전드", "웃음", "웃겼", "눈물", "entertainment", "celebrity", "girl group", "boy group", "idol", "actor", "singer", "variety show", "drama", "movie", "fan meeting", "comeback", "album", "music video", "teaser", "photoshoot", "dating", "breakup", "divorce", "marriage", "childbirth", "baseball", "soccer", "basketball", "volleyball", "golf", "esports", "K League", "KBO", "Premier League", "Champions League", "murder", "killing", "assault", "sexual assault", "rape", "kidnapping", "death", "corpse", "police", "arrest", "detention", "trial", "prison sentence", "lawsuit", "restaurant", "cafe", "tour spot", "travel diary", "tourism", "holiday", "weather", "fine dust", "traffic control", "shock", "scandal", "caught on camera", "backlash", "controversy", "reason why", "latest update", "netizens", "argument", "eventually", "identity", "disaster", "huge", "warning", "legendary", "funny", "laughter", "tearful", "읍사무소", "면사무소", "마을회관", "체험 행사", "지역 소식", "전통시장", "지역주민", "마을 주민", "농촌 체험", "어촌 체험", "지역 축제", "군민", "공모 사업"]
SOURCE_TIER_A = {"Reuters", "Bloomberg", "Financial Times", "The Wall Street Journal", "연합뉴스", "한국경제", "매일경제", "서울경제"}
SOURCE_TIER_B = {"중앙일보", "동아일보", "한겨레", "경향신문", "머니투데이", "전자신문", "ZDNet Korea", "TechCrunch", "The Verge"}
STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "with",
    "is", "are", "must", "should", "how", "become", "show", "little"
}

IMPACT_SIGNALS_MAP = {
    "policy": ["regulation", "rule", "policy", "bill", "law", "guideline", "government", "규제", "법안", "정책", "가이드라인", "정부", "국회"],
    "budget": ["budget", "fiscal", "appropriation", "incentive", "subsidy", "예산", "재정", "지원금", "세제혜택"],
    "sanctions": ["sanction", "export control", "entity list", "tariff", "제재", "수출통제", "블랙리스트", "관세"],
    "capex": ["data center", "datacentre", "capex", "investment", "build", "expansion", "infrastructure", "facility", "데이터센터", "증설", "투자", "설비"],
    "earnings": ["earnings", "guidance", "profit", "loss", "revenue", "흑자", "적자", "실적", "가이던스", "매출", "영업이익"],
    "market-demand": ["registrations", "registration", "deliveries", "delivery", "sales", "demand", "shipments", "등록", "판매", "수요"],
    "security": ["breach", "exploit", "ransomware", "cve", "vulnerability", "침해", "해킹", "랜섬웨어", "취약점"],
    "infra": ["outage", "downtime", "disruption", "장애", "정전", "서비스 중단"]
}

DEDUPE_NOISE_WORDS = {
    "bold", "little", "recovery", "shock", "inside", "first", "new", "top", "best",
    "strategy", "how", "why", "what", "where", "when", "show", "showcase", "unveils",
    "exclusive", "breaking", "update", "latest", "years", "after", "cornerstone", "become",
    "reuters", "bloomberg", "ft", "wsj", "financial", "times", "wall", "street", "journal",
    "연합뉴스", "매일경제", "한국경제", "서울경제", "머니투데이", "중앙일보", "동아일보",
    "한겨레", "경향신문", "techcrunch", "verge"
}

EMOTIONAL_DROP_KEYWORDS = ["참사", "충격", "분노", "논란", "폭로"]
DROP_CATEGORIES = {"사회", "사건", "연예"}

MONTH_TOKENS = {
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june",
    "jul", "july", "aug", "august", "sep", "sept", "september", "oct", "october", "nov", "november",
    "dec", "december"
}

LONG_IMPACT_SIGNALS = {"policy", "budget", "sanctions"}
MEDIA_SUFFIXES = ("일보", "신문", "뉴스", "방송", "미디어", "tv", "TV")

NEWSLETTER_TITLE = "🚀 DAILY WORLD – AI & Tech 일일 요약"
AFFILIATE_AD_TEXT = "🔥 오늘만 50% 할인! 최고의 생산성 도구 구경하기"
AFFILIATE_LINK = "https://your-affiliate-link.com"
OUTPUT_FILENAME = "daily_world_news.html"
OUTPUT_JSON = "daily_digest.json"
SELECTION_CRITERIA = "① 내일도 영향이 남는 이슈 ② 과도한 감정 소모 제외 ③ 어제와 중복되는 뉴스 제외"
EDITOR_NOTE = "이 뉴스는 클릭 수가 아니라 오늘 이후에도 남는 정보만 기준으로 편집했습니다."
QUESTION_OF_THE_DAY = "정보를 덜 보는 것이 오히려 더 똑똑한 소비일까?"

TOP_LIMIT = 5
MIN_SCORE = 0.0
MAX_ENTRIES_PER_FEED = 100

# ==========================================
# 핵심 로직 함수
# ==========================================

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

def pick_top_with_mix(all_items, top_limit=5):
    buckets = {"IT": [], "경제": [], "글로벌": []}
    for it in all_items:
        buckets[map_topic_to_category(it.get("topic", ""))].append(it)

    for cat in buckets:
        buckets[cat].sort(key=lambda x: x["score"], reverse=True)

    target = {"IT": 2, "경제": 2, "글로벌": 1}
    picked = []
    for cat, n in target.items():
        picked += buckets[cat][:n]

    if len(picked) < top_limit:
        remain = [x for x in sorted(all_items, key=lambda x: x["score"], reverse=True) if x not in picked]
        picked += remain[: top_limit - len(picked)]

    return picked[:top_limit]

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
    return dedupe_map

def fetch_news_grouped_and_top(sources, top_limit=3):
    print("🔍 뉴스를 수집하고 큐레이팅하는 중입니다...")
    grouped_items, seen_titles, all_items, topic_limits = {}, set(), [], {}
    seen_title_tokens: list[tuple[set[str], dict]] = []
    seen_items_by_dedupe_key = {}
    yesterday_dedupe_map = _load_yesterday_dedupe_map(OUTPUT_JSON)

    for source in sources:
        topic, url, feed_limit = source["topic"], source["url"], source.get("limit", 3)
        topic_limits[topic] = max(topic_limits.get(topic, 0), feed_limit)
        feed = feedparser.parse(url)
        
        for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
            title = getattr(entry, "title", "").strip()
            summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            source_name = get_source_name(entry)
            summary_clean = clean_text(summary_raw)
            summary_clean = _strip_source_from_text(summary_clean, source_name)
            title_clean = trim_title_noise(clean_text(title), source_name)
            summary = (summary_clean[:200] + "...") if summary_clean else "내용을 확인하려면 클릭하세요."

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
                "published": getattr(entry, "published", None), "score": score,
                "topic": topic, "source": source_name,
                "impactSignals": impact_signals, "dedupeKey": dedupe_key, "matchedTo": matched_to,
                "readTimeSec": read_time_sec
            }
            seen_title_tokens.append((tokens, item))
            seen_items_by_dedupe_key[dedupe_key] = item
            grouped_items.setdefault(topic, []).append(item)
            all_items.append(item)

    for topic, items in grouped_items.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        grouped_items[topic] = items[:topic_limits.get(topic, TOP_LIMIT)]

    return grouped_items, pick_top_with_mix(all_items, top_limit)

def main():
    try:
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
        print(f"✅ 완료! {OUTPUT_JSON} 파일이 생성되었습니다.")

    except Exception as e:
        print("❌ 오류 발생:", e)

if __name__ == "__main__":
    main()
