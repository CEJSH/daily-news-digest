import feedparser
import datetime
import webbrowser
import os
import re
import html
import json
from urllib.parse import urlparse
from jinja2 import Template


_WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not s:
        return ""
    # 1) &nbsp; 같은 HTML 엔티티를 문자로 변환
    s = html.unescape(s)

    # 2) NBSP(유니코드) -> 일반 스페이스로
    s = s.replace("\u00a0", " ")

    # 3) 혹시 섞여 들어온 HTML 태그 제거
    s = re.sub(r"<[^>]+>", "", s)

    # 4) 공백 정리
    s = _WS_RE.sub(" ", s).strip()
    return s

# ==========================================
# 사용자 설정
# ==========================================

RSS_SOURCES = [
    # ==========================
    # 로봇 (KR + Global)
    # ==========================
    {
        "topic": "로봇",
        "url": "https://news.google.com/rss/search?q=로봇&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "로봇",
        "url": "https://news.google.com/rss/search?q=robotics+OR+robot&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # AGI / 고급 AI (KR + Global)
    # ==========================
    {
        "topic": "AGI / 고급 AI",
        "url": "https://news.google.com/rss/search?q=AGI&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "AGI / 고급 AI",
        "url": "https://news.google.com/rss/search?q=AGI&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # AI / 인공지능 (KR + Global)
    # ==========================
    {
        "topic": "AI / 인공지능",
        "url": "https://news.google.com/rss/search?q=AI&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "AI / 인공지능",
        "url": "https://news.google.com/rss/search?q=AI&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # 반도체 (KR + Global)
    # ==========================
    {
        "topic": "반도체",
        "url": "https://news.google.com/rss/search?q=반도체&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "반도체",
        "url": "https://news.google.com/rss/search?q=semiconductor&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # 태양광 / 에너지 전환 (KR + Global)
    # ==========================
    {
        "topic": "태양광 / 에너지 전환",
        "url": "https://news.google.com/rss/search?q=태양광&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "태양광 / 에너지 전환",
        "url": "https://news.google.com/rss/search?q=solar+energy+OR+renewable+energy&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # 바이오 / 헬스케어 (KR + Global)
    # ==========================
    {
        "topic": "바이오 / 헬스케어",
        "url": "https://news.google.com/rss/search?q=바이오+헬스케어&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "바이오 / 헬스케어",
        "url": "https://news.google.com/rss/search?q=bio+healthcare+biotech&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },

    # ==========================
    # 규제 / 법·정책 (현재는 한국 위주)
    # ==========================
    {
        "topic": "규제 / 법·정책",
        "url": "https://news.google.com/rss/search?q=규제&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },

    # ==========================
    # 청년 (한국 이슈 위주)
    # ==========================
    {
        "topic": "서울",
        "url": "https://news.google.com/rss/search?q=서울&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },

        # ==========================
    # 청년 (한국 이슈 위주)
    # ==========================
    {
        "topic": "고용",
        "url": "https://news.google.com/rss/search?q=고용&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },

    # ==========================
    # 금융 / 자본시장 (KR + Global 예시)

    # ==========================
    {
        "topic": "금융 / 자본시장",
        "url": "https://news.google.com/rss/search?q=금융+자본시장&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 3,
    },
    {
        "topic": "금융 / 자본시장",
        "url": "https://news.google.com/rss/search?q=finance+capital+market&hl=en&gl=US&ceid=US:en",
        "limit": 3,
    },
]

USE_AI_SUMMARY = False  # 나중에 True로 바꿔서 활성화

def generate_ai_summary(title: str, summary: str, topic: str) -> str:
    """
    (선택) AI 요약을 생성하는 자리.
    - 지금은 빈 문자열 반환
    - 나중에 OpenAI 등 붙일 때 이 함수 안만 구현.
    """
    if not USE_AI_SUMMARY:
        return ""

    # 예시 (나중에 실제 API 붙일 때 사용)
    """
    import openai
    prompt = f'''
    제목: {title}
    주제: {topic}
    내용 요약: {summary}
    
    위 기사를 한국어로 2~3줄로 핵심만 짧게 요약해줘.
    '''
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
    """
    raise NotImplementedError(
        "USE_AI_SUMMARY=True 이지만 AI 요약 기능이 아직 구현되지 않았습니다."
    )

def generate_key_terms(title: str, summary: str, topic: str) -> list[str]:
    """
    기사 제목/요약/토픽을 기반으로 핵심 키워드 리스트를 생성하는 자리.
    - 지금은 USE_AI_KEYWORDS=False라서 항상 [] 반환
    - 나중에 OpenAI 같은 LLM 붙일 때 이 함수 안만 구현하면 됨.
    """
    if not USE_AI_KEYWORDS:
        return []

    # 아래는 실제 사용 시 구조 예시 (지금은 주석 처리용)
    """
    import openai

    prompt = f'''
    다음 뉴스 기사에서 공부/시장 분석에 중요한 핵심 키워드를 3~7개 뽑아줘.
    - 너무 일반적인 단어(뉴스, 오늘, 보도, 기자 등)는 제외.
    - 기술, 산업, 기업, 국가, 정책, 규제, 개념 단위 위주로.
    - 한국어/영어 혼용 가능. 각 키워드는 한두 단어 길이로.
    - 쉼표(,)로 구분해 한 줄로만 출력.

    [토픽] {topic}
    [제목] {title}
    [요약] {summary}
    '''

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    """

    raw = ""  # ↑ 나중에 실제 LLM 응답 문자열 넣을 자리

    if not raw:
        return []

    # "AGI, 규제, 미국, 반도체" → ["AGI", "규제", "미국", "반도체"]
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    return terms


NEWSLETTER_TITLE = "🚀 DAILY WORLD – AI & Tech 일일 요약"

AFFILIATE_AD_TEXT = "🔥 오늘만 50% 할인! 최고의 생산성 도구 구경하기"
AFFILIATE_LINK = "https://your-affiliate-link.com"

OUTPUT_FILENAME = "daily_world_news.html"

# ===== MVP JSON 출력 설정 =====
OUTPUT_JSON = "daily_digest.json"
# Lovable 화면 상단에 노출할 고정 문구들
SELECTION_CRITERIA = "① 내일도 영향이 남는 이슈 ② 과도한 감정 소모 제외 ③ 어제와 중복되는 뉴스 제외"
EDITOR_NOTE = "이 뉴스는 클릭 수가 아니라 오늘 이후에도 남는 정보만 기준으로 편집했습니다."
QUESTION_OF_THE_DAY = "정보를 덜 보는 것이 오히려 더 똑똑한 소비일까?"

# 키워드 자동 생성(LLM) 기능 토글 (현재는 비활성 권장)
USE_AI_KEYWORDS = False


TOP_LIMIT = 5  # 전체 TOP N (MVP: 5개 고정)
# topic별 최대 기사 개수는 각 topic에 설정된 limit 중 최대값을 사용 (뒤에서 계산)


# ==========================================
# 큐레이션 기준 (여기 위주로 튜닝)
# ==========================================



QUALITY_KEYWORDS = [
    "분석", "해설", "전망", "심층", "진단",
    "전략", "패권", "패러다임", "변곡점", "구조", "재편", "지형",
    "모멘텀", "구조적", "생태계", "시나리오",
    "data", "in-depth", "diagnosis", "strategy", "paradigm",
    "inflection point", "structure", "reorganization", "ecosystem", "scenario",
]


HARD_EXCLUDE_KEYWORDS = [
    # 리포트/기관/홍보/행사/모집
    "동향", "동향리포트", "리포트", "브리프", "백서", "자료집", "보고서", "연구보고서",
    "세미나", "웨비나", "컨퍼런스", "포럼", "행사", "모집", "신청", "접수",
    "보도자료", "홍보", "프로모션", "할인", "출시기념",
    # 영문
    "whitepaper", "report", "brief", "webinar", "conference", "forum",
    "press release", "promotion", "apply now",
]

HARD_EXCLUDE_URL_HINTS = [
    "/report", "/whitepaper", "/webinar", "/seminar", "/conference", "/event", "/download"
]


EXCLUDE_KEYWORDS = [
    # 연예/가십
    "연예", "스타", "걸그룹", "보이그룹", "아이돌",
    "배우", "가수", "예능", "드라마", "영화", "팬미팅",
    "컴백", "앨범", "뮤직비디오", "뮤비", "티저", "화보",
    "열애", "결별", "이혼", "결혼", "출산",

    # 스포츠
    "야구", "축구", "농구", "배구", "골프", "e스포츠",
    "K리그", "KBO", "프리미어리그", "챔피언스리그",

    # 사건사고(치명적인 범죄/자극적 보도)
    "살해", "살인", "폭행", "성폭행", "강간", "납치",
    "사망", "시신",  "징역", 

    # 너무 로컬한 생활/가십
    "맛집", "카페", "뷰맛집", "여행기", "관광지", "연휴",
    "날씨", "미세먼지", "교통통제",

    # 그 외 (자극/클릭베이트)
    "경악", "발칵", "알고보니", "이유는", "근황",
    "포착", "망신", "누리꾼", "갑론을박", "결국", "정체", "충격", "헉", "소름", "이게 얼마", "대참사", "대박",
    "주의보", "레전드", "웃음", "웃겼", "눈물",

    # Entertainment / Gossip (영문)
    "entertainment", "celebrity", "girl group", "boy group", "idol",
    "actor", "singer", "variety show", "drama", "movie", "fan meeting",
    "comeback", "album", "music video", "teaser", "photoshoot",
    "dating", "breakup", "divorce", "marriage", "childbirth",

    # Sports (영문)
    "baseball", "soccer", "basketball", "volleyball", "golf", "esports",
    "K League", "KBO", "Premier League", "Champions League",

    # Crime / Sensational Incidents (영문)
    "murder", "killing", "assault", "sexual assault", "rape", "kidnapping",
    "death", "corpse", "police", "arrest", "detention",
    "trial", "prison sentence", "lawsuit",

    # Local lifestyle/gossip (영문)
    "restaurant", "cafe", "tour spot", "travel diary", "tourism", "holiday",
    "weather", "fine dust", "traffic control",

    # Sensational / Clickbait (영문)
    "shock", "scandal", "caught on camera", "backlash", "controversy",
    "reason why", "latest update", "netizens", "argument", "eventually",
    "identity", "disaster", "huge", "warning", "legendary",
    "funny", "laughter", "tearful",

     # 지역/생활/행사/공모 등 
   "읍사무소", "면사무소", "마을회관", "체험 행사", "지역 소식",
   "전통시장", "지역주민", "마을 주민",
    "농촌 체험", "어촌 체험", "지역 축제", "군민",
    "공모 사업", 
]

SOURCE_TIER_A = {"Reuters", "Bloomberg", "Financial Times", "The Wall Street Journal", "연합뉴스", "한국경제", "매일경제", "서울경제"}
SOURCE_TIER_B = {"중앙일보", "동아일보", "한겨레", "경향신문", "머니투데이", "전자신문", "ZDNet Korea", "TechCrunch", "The Verge"}


MIN_SCORE = 2.0
MAX_ENTRIES_PER_FEED = 100

# HTML 태그 제거용 정규식
TAG_RE = re.compile(r"<[^>]+>")


# ==========================================
# 유틸리티 함수
# ==========================================
def pick_top_with_mix(all_items, top_limit=5):
    buckets = {"IT": [], "경제": [], "글로벌": []}
    for it in all_items:
        cat = map_topic_to_category(it.get("topic", ""))
        buckets[cat].append(it)

    for cat in buckets:
        buckets[cat].sort(key=lambda x: x["score"], reverse=True)

    target = {"IT": 2, "경제": 2, "글로벌": 1}
    picked = []
    for cat, n in target.items():
        picked += buckets[cat][:n]

    # 부족하면 전체에서 추가
    if len(picked) < top_limit:
        remain = [x for x in sorted(all_items, key=lambda x: x["score"], reverse=True) if x not in picked]
        picked += remain[: top_limit - len(picked)]

    return picked[:top_limit]


def source_weight(source_name: str) -> float:
    if source_name in SOURCE_TIER_A:
        return 3.0
    if source_name in SOURCE_TIER_B:
        return 1.5
    return 0.3


def trim_title_noise(title: str) -> str:
    # 너무 공격적이면 위험하니, 우선 ' | ' 한 번만 컷
    return title.split(" | ")[0].strip()

def get_source_name(entry) -> str:
    """Google News RSS에서 언론사 이름(source.title)을 가져옴."""
    try:
        if hasattr(entry, "source") and hasattr(entry.source, "title"):
            return entry.source.title.strip()
    except Exception:
        pass
    return ""


def score_entry(entry) -> float:
    """
    RSS entry 하나에 대해 '양질 + 구조적 중요도' 점수 계산.
    - 언론사 이름
    - 인사이트 키워드
    - 연예/가십/사건사고 필터링 (하드 필터)
    - 최신성
    - 요약 길이
    """
    score = 0.0

    


    title_raw = getattr(entry, "title", "") or ""
    summary_raw = getattr(entry, "summary", "") or ""

    if "|" in title_raw or ">" in title_raw or "…" in title_raw or "..." in title_raw:
        score -= 1.0

    title = trim_title_noise(clean_text(title_raw))
    summary = clean_text(summary_raw)
    source_name = get_source_name(entry)

    link = getattr(entry, "link", "") or ""
    text_all = (title + " " + summary).lower()

    for bad in HARD_EXCLUDE_KEYWORDS:
        if bad.lower() in text_all:
            return -999.0

    low_link = link.lower()
    for hint in HARD_EXCLUDE_URL_HINTS:
        if hint in low_link:
            return -999.0

    # 0) 연예/가십/사건사고 등은 아예 제외 (하드 필터)
    for bad in EXCLUDE_KEYWORDS:
        if bad.lower() in text_all:
            return -999.0  # MIN_SCORE보다 훨씬 작게 → 무조건 버림
    

    # 1) 언론사 신뢰도
    score += source_weight(source_name)

    # 2) 인사이트/분석 키워드 가점
    quality_hits = 0
    for kw in QUALITY_KEYWORDS:
        if kw.lower() in text_all:
            quality_hits += 1

    score += min(quality_hits, 2) * 1.0   # 최대 2개까지만, 가중치도 낮춤


    # 3) 제목이 너무 짧으면 감점
    if len(title) < 10:
        score -= 0.5

    # 4) 요약 길이 (너무 짧으면 감점)
    if len(summary) < 40:
        score -= 0.5

    # 5) 최신성 (published 기준)
    published_parsed = getattr(entry, "published_parsed", None)
    if published_parsed:
        published_dt = datetime.datetime(*published_parsed[:6])
        now = datetime.datetime.now()
        delta = now - published_dt

        if delta.days < 1:
            score += 1.3  # 24시간 이내
        elif delta.days < 3:
            score += 1.0  # 3일 이내
        elif delta.days < 7:
            score += 0.7  # 7일 이내
        elif delta.days > 21:
            score -= 1.0  # 3주 이상 지난 글은 감점

    return score


# ==========================================
# 뉴스 수집 및 가공
# ==========================================

def fetch_news_grouped_and_top(sources, top_limit=3):
    """
    - 주제별(grouped_items)로 필터링/스코어 적용된 뉴스 모음
    - 전체 기사 중 TOP N (top_items)

    같은 topic을 쓰는 여러 RSS 소스(KR/EN 등)를 모두 합쳐서
    topic 단위로 정렬 후 상위 limit개만 남긴다.
    """
    print("🔍 뉴스를 수집하고 큐레이팅하는 중입니다...")

    grouped_items = {}       # topic -> [item, item, ...]
    seen_titles = set()      # 전체 중복 제거
    all_items = []           # 전체 기사 모음
    topic_limits = {}        # topic별 limit 설정 (같은 topic의 여러 소스 중 최대값 사용)

    for source in sources:
        topic = source["topic"]
        url = source["url"]
        feed_limit = source.get("limit", 3)

        # topic별 limit 재정의: 같은 topic이 여러 소스에 걸쳐 있으면, limit의 최대값을 사용
        topic_limits[topic] = max(topic_limits.get(topic, 0), feed_limit)

        feed = feedparser.parse(url)

        # 너무 많은 기사 방지
        entries = feed.entries[:MAX_ENTRIES_PER_FEED]

        for entry in entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
            summary_clean = clean_text(summary_raw)
            summary = (summary_clean[:200] + "...") if summary_clean else "내용을 확인하려면 클릭하세요."

            if not title:
                continue

            # 제목 기준 전역 중복 제거
            if title in seen_titles:
                continue

            score = score_entry(entry)

            # 최소 점수 미만이면 아예 버림
            if score < MIN_SCORE:
                continue

            seen_titles.add(title)
            published = getattr(entry, "published", None)
            source_name = get_source_name(entry)

            item = {
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
                "score": score,
                "topic": topic,
                "source": source_name,
            }

            # topic별로 누적
            if topic not in grouped_items:
                grouped_items[topic] = []
            grouped_items[topic].append(item)

            all_items.append(item)

    # topic별로 점수 순 정렬 후 topic별 limit까지 자르기
    for topic, items in grouped_items.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        limit_for_topic = topic_limits.get(topic, TOP_LIMIT)
        grouped_items[topic] = items[:limit_for_topic]


    top_items = pick_top_with_mix(all_items, top_limit)

    return grouped_items, top_items



# ==========================================
# MVP JSON 내보내기 (Lovable/SPA 소비용)
# ==========================================

def map_topic_to_category(topic: str) -> str:
    """현재 RSS topic을 MVP 3카테고리(IT/경제/글로벌)로 매핑."""
    t = (topic or "").lower()
    it_keywords = ["ai", "agi", "로봇", "robot", "반도체", "semiconductor", "인공지능"]
    econ_keywords = ["경제", "finance", "금리", "환율", "주가", "증시", "투자", "에너지", "태양광", "energy"]

    if any(k in t for k in it_keywords):
        return "IT"
    if any(k in t for k in econ_keywords):
        return "경제"
    return "글로벌"


def split_summary_to_3lines(summary: str) -> list[str]:
    """요약 문자열을 최대 3줄 배열로 변환. (MVP UI용)"""
    s = (summary or "").strip()
    if not s:
        return []

    # 문장 단위 분리(영문/국문 공통) → 최대 3개
    parts = [p.strip() for p in re.split(r'(?<=[\.\!\?。])\s+|(?<=다\.)\s+', s) if p.strip()]
    if len(parts) >= 3:
        return parts[:3]

    # 문장 분리가 애매하면 길이로 균등 분할
    if len(parts) <= 1 and len(s) > 120:
        step = max(40, len(s)//3)
        chunks = [s[i:i+step].strip() for i in range(0, len(s), step)]
        return chunks[:3]

    return parts


def estimate_read_time_seconds(text: str) -> int:
    """한국어 평균 읽기 속도 ~500자/분 가정. 10초 단위 반올림, 10~40초로 클램프."""
    n = len((text or "").strip())
    if n <= 0:
        return 10
    seconds = (n / 500) * 60
    # 10초 단위 반올림
    rounded = int(round(seconds / 10) * 10)
    return max(10, min(40, rounded))


def _load_existing_digest(path: str = OUTPUT_JSON) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_valid_digest(digest: dict) -> bool:
    """MVP 안전장치: 5개 고정 + 핵심 필드 존재 여부만 검사 (엄격하게)."""
    if not isinstance(digest, dict):
        return False
    items = digest.get("items")
    if not isinstance(items, list) or len(items) != 5:
        return False

    required_item_keys = {"id", "date", "category", "title", "summary", "sourceName", "sourceUrl", "status", "importance"}
    for it in items:
        if not isinstance(it, dict):
            return False
        if not required_item_keys.issubset(it.keys()):
            return False
        if not it.get("title") or not it.get("sourceUrl"):
            return False
        summary = it.get("summary")
        if not isinstance(summary, list) or len(summary) == 0:
            return False
    return True


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def export_daily_digest_json(top_items: list[dict], output_path: str = OUTPUT_JSON) -> dict:
    """fetch_news_grouped_and_top()의 top_items를 MVP 스키마로 변환해 JSON으로 저장.
    RSS 장애/예외로 5개를 못 채우면, 기존 JSON을 유지한다(있을 때).
    """
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now_kst.strftime("%Y-%m-%d")
    last_updated_at = now_kst.isoformat()

    items_out: list[dict] = []
    for i, item in enumerate(top_items[:5], start=1):
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = (item.get("summary") or "").strip()
        topic = (item.get("topic") or "").strip()
        source_name = (item.get("source") or "").strip()
        published = item.get("published")

        summary_lines = split_summary_to_3lines(summary)
        read_time_sec = estimate_read_time_seconds(" ".join(summary_lines) if summary_lines else summary)

        items_out.append(
            {
                "id": f"{date_str}_{i}",
                "date": date_str,
                "category": map_topic_to_category(topic),
                "title": title,
                "summary": summary_lines if summary_lines else [summary],
                "whyImportant": "",  # MVP: 수동 입력 권장 (서비스 차별화 핵심)
                "sourceName": source_name,
                "sourceUrl": link,
                "publishedAt": published,
                "readTimeSec": read_time_sec,
                "status": "published",
                "importance": 1,
            }
        )

    digest = {
        "date": date_str,
        "selectionCriteria": SELECTION_CRITERIA,
        "editorNote": EDITOR_NOTE,
        "question": QUESTION_OF_THE_DAY,
        "lastUpdatedAt": last_updated_at,
        "items": items_out,
    }

    if not _is_valid_digest(digest):
        existing = _load_existing_digest(output_path)
        if existing and _is_valid_digest(existing):
            print("⚠️ 오늘 digest 생성이 불완전하여 기존 daily_digest.json을 유지합니다.")
            return existing
        raise RuntimeError("digest 생성 실패: 유효한 5개 뉴스가 생성되지 않았고 기존 파일도 없습니다.")

    _atomic_write_json(output_path, digest)
    return digest


# ==========================================
# HTML 생성
# ==========================================

def generate_html(grouped_items, top_items):
    print("📝 HTML 뉴스레터를 생성하는 중입니다...")

    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8" />
        <title>{{ title }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
                background-color: #f4f4f4;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: #ffffff;
                padding: 28px;
                border-radius: 12px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
            }
            h1 {
                color: #1f2933;
                text-align: center;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 16px;
                margin-top: 0;
                margin-bottom: 8px;
            }
            .date {
                text-align: center;
                color: #9ca3af;
                font-size: 13px;
                margin-bottom: 20px;
            }
            .intro {
                font-size: 14px;
                color: #4b5563;
                line-height: 1.6;
                margin-bottom: 24px;
            }

            /* TOP 3 섹션 */
            .top-section {
                margin-bottom: 28px;
                padding: 16px;
                border-radius: 10px;
                background: #f9fafb;
                border: 1px solid #e5e7eb;
            }
            .top-section-title {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 12px;
            }
            .top-list {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
            }
            @media (min-width: 720px) {
                .top-list {
                    grid-template-columns: 1fr 1fr;
                }
            }
            .top-item {
                padding: 12px 14px;
                border-radius: 10px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
            }
            .top-rank {
                font-size: 12px;
                font-weight: 700;
                color: #2563eb;
                margin-bottom: 4px;
            }
            .top-topic {
                font-size: 11px;
                color: #6b7280;
                margin-bottom: 2px;
            }
            .top-source {
                font-size: 11px;
                color: #9ca3af;
                margin-bottom: 4px;
            }
            .top-title {
                font-size: 15px;
                font-weight: 600;
                color: #111827;
                text-decoration: none;
            }
            .top-title:hover {
                text-decoration: underline;
            }
            .top-summary {
                margin-top: 6px;
                font-size: 13px;
                color: #4b5563;
                line-height: 1.5;
            }
            .top-published {
                margin-top: 4px;
                font-size: 11px;
                color: #9ca3af;
            }

            /* 주제별 섹션 */
            .topic-section {
                margin-top: 24px;
                margin-bottom: 12px;
                padding-top: 12px;
                border-top: 1px solid #e5e7eb;
            }
            .topic-title {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 10px;
            }
            .news-item {
                margin-bottom: 18px;
            }
            .news-title {
                font-size: 14px;
                font-weight: 600;
                color: #2563eb;
                text-decoration: none;
            }
            .news-title:hover {
                text-decoration: underline;
            }
            .news-summary {
                color: #4b5563;
                font-size: 13px;
                margin-top: 4px;
                line-height: 1.5;
            }
            .published {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 3px;
            }
            .source {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 2px;
            }

            .ad-block {
                background-color: #fff7ed;
                border: 1px solid #fed7aa;
                color: #9a3412;
                padding: 16px;
                text-align: center;
                margin-top: 32px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            .ad-link {
                text-decoration: none;
                color: #dc2626;
            }
            .ad-link:hover {
                text-decoration: underline;
            }
            .footer {
                text-align: center;
                font-size: 11px;
                color: #9ca3af;
                margin-top: 24px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{ title }}</h1>
            <div class="date">{{ date }}</div>

            <div class="intro">
                오늘 세계 흐름을 읽는 데 중요한
                <strong>AI · 반도체 · 에너지 · 바이오 · 규제 · 금융</strong> 뉴스를
                한 번에 모았습니다. 맨 위에는 강화된 기준으로 선별한
                <strong>TOP {{ top_count }} 핵심 뉴스</strong>가, 그 아래에는
                주제별 섹션이 이어집니다.
            </div>

            {% if top_items %}
            <div class="top-section">
                <div class="top-section-title">🔥 오늘의 핵심 TOP {{ top_count }}</div>
                <div class="top-list">
                    {% for item in top_items %}
                    <div class="top-item">
                        <div class="top-rank">TOP {{ loop.index }}</div>
                        <div class="top-topic">{{ item.topic }}</div>
                        {% if item.source %}
                        <div class="top-source">{{ item.source }}</div>
                        {% endif %}
                        <a href="{{ item.link }}" target="_blank" class="top-title">{{ item.title }}</a>
                        {% if item.published %}
                        <div class="top-published">{{ item.published }}</div>
                        {% endif %}
                        <div class="top-summary">{{ item.summary }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% for topic, items in grouped_items.items() %}
            <div class="topic-section">
                <div class="topic-title">📌 {{ topic }}</div>
                {% for item in items %}
                    <div class="news-item">
                        <a href="{{ item.link }}" class="news-title" target="_blank">👉 {{ item.title }}</a>
                        {% if item.source %}
                        <div class="source">{{ item.source }}</div>
                        {% endif %}
                        {% if item.published %}
                        <div class="published">{{ item.published }}</div>
                        {% endif %}
                        <p class="news-summary">{{ item.summary }}</p>
                    </div>
                {% endfor %}
            </div>
            {% endfor %}

            <div class="ad-block">
                <a href="{{ ad_link }}" class="ad-link" target="_blank">{{ ad_text }}</a>
            </div>

            <div class="footer">
                Automated by DAILY WORLD v1.0<br />
                이 페이지는 개인용 자동 뉴스 요약 봇이 생성했습니다.
            </div>
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    today = datetime.datetime.now().strftime("%Y년 %m월 %d일 (%a)")

    return template.render(
        title=NEWSLETTER_TITLE,
        date=today,
        grouped_items=grouped_items,
        top_items=top_items,
        top_count=len(top_items),
        ad_text=AFFILIATE_AD_TEXT,
        ad_link=AFFILIATE_LINK,
    )


# ==========================================
# 메인
# ==========================================



def main():
    try:
        grouped_items, top_items = fetch_news_grouped_and_top(
            RSS_SOURCES, top_limit=TOP_LIMIT
        )

        # 1) MVP용 JSON 생성 (Lovable/SPA에서 바로 사용)
        export_daily_digest_json(top_items, OUTPUT_JSON)
        print(f"✅ 완료! {OUTPUT_JSON} 파일이 생성되었습니다.")

        # 2) (선택) 기존 HTML 뉴스레터도 계속 쓰고 싶다면 아래 주석을 해제
        # html_content = generate_html(grouped_items, top_items)
        # with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        #     f.write(html_content)
        # print(f"✅ 완료! {OUTPUT_FILENAME} 파일이 생성되었습니다.")
        # file_url = "file://" + os.path.realpath(OUTPUT_FILENAME)
        # webbrowser.open(file_url)

    except Exception as e:
        print("❌ 오류 발생:", e)


if __name__ == "__main__":
    main()
