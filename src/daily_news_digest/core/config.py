from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from daily_news_digest.core.constants import SOURCE_TIER_A, SOURCE_TIER_B

if load_dotenv:
    _repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=_repo_root / ".env")

# ==========================================
# 사용자 설정 (수정 가능)
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
     # 1) 기업 실적·가이던스
    {
        "topic": "실적_가이던스",
        "url": "https://news.google.com/rss/search?q=실적+OR+가이던스+OR+전망+OR+매출+OR+영업이익+OR+컨센서스+-칼럼+-리포트+-세미나&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 15
    },
    {
        "topic": "실적_가이던스",
        "url": "https://news.google.com/rss/search?q=earnings+OR+guidance+OR+forecast+OR+quarterly+results+OR+revenue+OR+margin+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 15
    },
    # 2) 반도체 공급망
    {
        "topic": "반도체_공급망",
        "url": "https://news.google.com/rss/search?q=HBM+OR+첨단패키징+OR+파운드리+OR+EUV+OR+반도체장비+OR+수출통제+-칼럼+-리포트+-세미나&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 15
    },
    {
        "topic": "반도체_공급망",
        "url": "https://news.google.com/rss/search?q=HBM+OR+advanced+packaging+OR+foundry+OR+EUV+OR+semiconductor+equipment+OR+export+controls+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 15
    },
    # 3) 전력 인프라
    {
        "topic": "전력_인프라",
        "url": "https://news.google.com/rss/search?q=전력망+OR+송전+OR+변전소+OR+전기요금+OR+원전+OR+LNG+OR+전력수급+-칼럼+-리포트+-연예+-스포츠&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 15
    },
    {
        "topic": "전력_인프라",
        "url": "https://news.google.com/rss/search?q=power+grid+OR+electricity+prices+OR+utility+OR+nuclear+OR+natural+gas+OR+transmission+OR+substation+OR+data+center+power+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 15
    },
    # 4) AI 저작권·데이터 권리
    {
        "topic": "AI_저작권_데이터권리",
        "url": "https://news.google.com/rss/search?q=AI+저작권+OR+학습데이터+OR+라이선스+OR+개인정보+OR+데이터보호+-칼럼+-리포트+-세미나&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 12
    },
    {
        "topic": "AI_저작권_데이터권리",
        "url": "https://news.google.com/rss/search?q=AI+copyright+OR+training+data+OR+licensing+OR+privacy+OR+data+protection+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 12
    },
    # 5) 보안 취약점·패치
    {
        "topic": "보안_취약점_패치",
        "url": "https://news.google.com/rss/search?q=취약점+OR+CVE+OR+제로데이+OR+보안패치+OR+권고+OR+침해사고+-칼럼+-연예+-스포츠&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 12
    },
    {
        "topic": "보안_취약점_패치",
        "url": "https://news.google.com/rss/search?q=zero-day+OR+patch+OR+CVE+OR+ransomware+OR+breach+notification+OR+incident+response+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 12
    },
    # 6) 투자·M&A·IPO
    {
        "topic": "투자_MA_IPO",
        "url": "https://news.google.com/rss/search?q=IPO+OR+상장+OR+인수합병+OR+투자유치+OR+시리즈A+OR+벤처캐피탈+-칼럼+-연예+-스포츠&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 12
    },
    {
        "topic": "투자_MA_IPO",
        "url": "https://news.google.com/rss/search?q=IPO+OR+acquisition+OR+merger+OR+funding+round+OR+venture+capital+-opinion+-column+-webinar&hl=en&gl=US&ceid=US:en",
        "limit": 12
    },
    # 7) 국내 정책·제도
    {
        "topic": "국내_정책_규제",
        "url": "https://news.google.com/rss/search?q=국회+OR+입법+OR+시행령+OR+가이드라인+OR+금융위원회+OR+공정거래위원회+OR+개인정보보호위원회+OR+과학기술정보통신부+-연예+-스포츠+-칼럼&hl=ko&gl=KR&ceid=KR:ko",
        "limit": 15
    }

]

NEWSLETTER_TITLE = "🚀 DAILY WORLD – AI & Tech 일일 요약"
AFFILIATE_AD_TEXT = "🔥 오늘만 50% 할인! 최고의 생산성 도구 구경하기"
AFFILIATE_LINK = "https://your-affiliate-link.com"

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))

OUTPUT_JSON = os.getenv("OUTPUT_JSON", str(DATA_DIR / "daily_digest.json"))
DEDUPE_HISTORY_PATH = os.getenv("DEDUPE_HISTORY_PATH", str(DATA_DIR / "dedupe_history.json"))
METRICS_JSON = os.getenv("METRICS_JSON", str(DATA_DIR / "digest_metrics.json"))
DEDUPE_RECENT_DAYS = int(os.getenv("DEDUPE_RECENT_DAYS", "3"))
SOURCE_WEIGHT_ENABLED = os.getenv("SOURCE_WEIGHT_ENABLED", "1") == "1"
SOURCE_WEIGHT_FACTOR = float(os.getenv("SOURCE_WEIGHT_FACTOR", "0.6"))
SELECTION_CRITERIA = "① 내일도 영향이 남는 이슈 ② 과도한 감정 소모 제외 ③ 어제와 중복되는 뉴스 제외"
EDITOR_NOTE = "이 뉴스는 클릭 수가 아니라 오늘 이후에도 남는 정보만 기준으로 편집했습니다."
QUESTION_OF_THE_DAY = "정보를 덜 보는 것이 오히려 더 똑똑한 소비일까?"

TOP_LIMIT = 20
MIN_TOP_ITEMS = int(os.getenv("MIN_TOP_ITEMS", "5"))
MIN_SCORE = 0.0
MAX_ENTRIES_PER_FEED = 150
TITLE_DEDUPE_JACCARD = float(os.getenv("TITLE_DEDUPE_JACCARD", "0.55"))
DEDUPKEY_NGRAM_N = int(os.getenv("DEDUPKEY_NGRAM_N", "2"))
DEDUPKEY_NGRAM_SIM = float(os.getenv("DEDUPKEY_NGRAM_SIM", "0.35"))

def _parse_csv_env(name: str) -> list[str]:
    """CSV 형태의 환경변수를 리스트로 파싱."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

def _env_int(name: str) -> int | None:
    """정수형 환경변수를 안전하게 파싱."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None

def _auto_tuned_limit(default_limit: int, multiplier: float, min_floor: int) -> int:
    """TOP_LIMIT 기준으로 상/하한을 지키며 자동 조정."""
    max_cap = max(default_limit, TOP_LIMIT)
    tuned = int(TOP_LIMIT * multiplier)
    tuned = max(min_floor, tuned)
    tuned = min(max_cap, tuned)
    return tuned


def _resolve_limit(
    env_name: str,
    default_limit: int,
    *,
    multiplier: float,
    min_floor: int,
    auto_tune: bool,
) -> int:
    """환경변수 우선, 없으면 자동 조정 또는 기본값 사용."""
    explicit = _env_int(env_name)
    if explicit is not None:
        return explicit
    if auto_tune:
        return _auto_tuned_limit(default_limit, multiplier, min_floor)
    return default_limit

# ==========================================
# 환경변수 기반 설정
# ==========================================

AI_IMPORTANCE_ENABLED = os.getenv("AI_IMPORTANCE_ENABLED", "1") == "1"
AI_AUTO_TUNE = os.getenv("AI_AUTO_TUNE", "1") == "1"
_DEFAULT_AI_IMPORTANCE_MAX = 200
_DEFAULT_AI_SEMANTIC_MAX = 50
AI_IMPORTANCE_MAX_ITEMS = _resolve_limit(
    "AI_IMPORTANCE_MAX_ITEMS",
    _DEFAULT_AI_IMPORTANCE_MAX,
    multiplier=1.2,
    min_floor=max(10, TOP_LIMIT),
    auto_tune=AI_AUTO_TUNE,
)
AI_IMPORTANCE_WEIGHT = float(os.getenv("AI_IMPORTANCE_WEIGHT", "1.0"))
AI_QUALITY_ENABLED = os.getenv("AI_QUALITY_ENABLED", "1") == "1"
AI_SEMANTIC_DEDUPE_ENABLED = os.getenv("AI_SEMANTIC_DEDUPE_ENABLED", "1") == "1"
AI_SEMANTIC_DEDUPE_MAX_ITEMS = _resolve_limit(
    "AI_SEMANTIC_DEDUPE_MAX_ITEMS",
    _DEFAULT_AI_SEMANTIC_MAX,
    multiplier=1.6,
    min_floor=max(12, TOP_LIMIT),
    auto_tune=AI_AUTO_TUNE,
)
AI_SEMANTIC_DEDUPE_THRESHOLD = float(os.getenv("AI_SEMANTIC_DEDUPE_THRESHOLD", "0.88"))
# 2차 패스 (key constraints 무시, 더 공격적): 1차보다 낮은 임계값으로 보강
AI_SEMANTIC_DEDUPE_THRESHOLD_RELAXED = float(
    os.getenv("AI_SEMANTIC_DEDUPE_THRESHOLD_RELAXED", "0.85")
)
ARTICLE_FETCH_ENABLED = os.getenv("ARTICLE_FETCH_ENABLED", "1") == "1"
ARTICLE_FETCH_MAX_ITEMS = int(os.getenv("ARTICLE_FETCH_MAX_ITEMS", "200"))
ARTICLE_FETCH_MIN_CHARS = int(os.getenv("ARTICLE_FETCH_MIN_CHARS", "300"))
ARTICLE_FETCH_TIMEOUT_SEC = int(os.getenv("ARTICLE_FETCH_TIMEOUT_SEC", "6"))
FULLTEXT_LOG_ENABLED = os.getenv("FULLTEXT_LOG_ENABLED", "0") == "1"
FULLTEXT_LOG_MAX_CHARS = int(os.getenv("FULLTEXT_LOG_MAX_CHARS", "50"))

# ==========================================
# TOP 20 품질 강화 옵션
# ==========================================

TOP_SOURCE_ALLOWLIST_ENABLED = os.getenv("TOP_SOURCE_ALLOWLIST_ENABLED", "1") == "1"
TOP_SOURCE_ALLOWLIST_STRICT = os.getenv("TOP_SOURCE_ALLOWLIST_STRICT", "1") == "1"
_allowlist_env = set(_parse_csv_env("TOP_SOURCE_ALLOWLIST"))
TOP_SOURCE_ALLOWLIST = _allowlist_env if _allowlist_env else (set(SOURCE_TIER_A) | set(SOURCE_TIER_B))

TOP_FRESH_MAX_HOURS = int(os.getenv("TOP_FRESH_MAX_HOURS", "48"))
_fresh_except_env = set(_parse_csv_env("TOP_FRESH_EXCEPT_SIGNALS"))
TOP_FRESH_EXCEPT_SIGNALS = _fresh_except_env if _fresh_except_env else {"policy", "sanctions", "earnings", "stats"}
TOP_FRESH_EXCEPT_MAX_HOURS = int(os.getenv("TOP_FRESH_EXCEPT_MAX_HOURS", "168"))
TOP_REQUIRE_PUBLISHED = os.getenv("TOP_REQUIRE_PUBLISHED", "1") == "1"

# ==========================================
# Signal cap (편향 완화)
# ==========================================

SIGNAL_CAP_ENABLED = os.getenv("SIGNAL_CAP_ENABLED", "1") == "1"
SIGNAL_CAP_RATIO = float(os.getenv("SIGNAL_CAP_RATIO", "0.3"))
SIGNAL_CAP_PENALTY = float(os.getenv("SIGNAL_CAP_PENALTY", "0.7"))
_signal_cap_labels_env = set(_parse_csv_env("SIGNAL_CAP_LABELS"))
SIGNAL_CAP_LABELS = _signal_cap_labels_env if _signal_cap_labels_env else {"policy", "sanctions", "earnings"}
SIGNAL_CAP_EXCEPT_LONG_TRIGGER = os.getenv("SIGNAL_CAP_EXCEPT_LONG_TRIGGER", "1") == "1"
SIGNAL_CAP_EXCEPT_IMPORTANCE = int(os.getenv("SIGNAL_CAP_EXCEPT_IMPORTANCE", "4"))

# ==========================================
# low_quality 정책
# - drop: qualityLabel=low_quality인 아이템은 status=dropped로 강제
# - downgrade: status는 유지하되 importance를 1 이하로 하향
# ==========================================

LOW_QUALITY_POLICY = (os.getenv("LOW_QUALITY_POLICY", "drop") or "drop").strip().lower()
if LOW_QUALITY_POLICY not in {"drop", "downgrade"}:
    LOW_QUALITY_POLICY = "drop"

LOW_QUALITY_DOWNGRADE_MAX_IMPORTANCE = int(os.getenv("LOW_QUALITY_DOWNGRADE_MAX_IMPORTANCE", "1"))
LOW_QUALITY_DOWNGRADE_RATIONALE = os.getenv(
    "LOW_QUALITY_DOWNGRADE_RATIONALE",
    "근거 부족이라 영향 판단 불가",
).strip()
