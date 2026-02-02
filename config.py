import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()

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
]

NEWSLETTER_TITLE = "🚀 DAILY WORLD – AI & Tech 일일 요약"
AFFILIATE_AD_TEXT = "🔥 오늘만 50% 할인! 최고의 생산성 도구 구경하기"
AFFILIATE_LINK = "https://your-affiliate-link.com"
OUTPUT_FILENAME = "daily_world_news.html"
OUTPUT_JSON = "daily_digest.json"
SELECTION_CRITERIA = "① 내일도 영향이 남는 이슈 ② 과도한 감정 소모 제외 ③ 어제와 중복되는 뉴스 제외"
EDITOR_NOTE = "이 뉴스는 클릭 수가 아니라 오늘 이후에도 남는 정보만 기준으로 편집했습니다."
QUESTION_OF_THE_DAY = "정보를 덜 보는 것이 오히려 더 똑똑한 소비일까?"

TOP_LIMIT = 20
MIN_SCORE = 0.0
MAX_ENTRIES_PER_FEED = 100

# ==========================================
# 환경변수 기반 설정
# ==========================================

AI_IMPORTANCE_ENABLED = os.getenv("AI_IMPORTANCE_ENABLED", "1") == "1"
AI_IMPORTANCE_MAX_ITEMS = int(os.getenv("AI_IMPORTANCE_MAX_ITEMS", "30"))
AI_IMPORTANCE_WEIGHT = float(os.getenv("AI_IMPORTANCE_WEIGHT", "1.0"))
AI_QUALITY_ENABLED = os.getenv("AI_QUALITY_ENABLED", "1") == "1"
AI_SEMANTIC_DEDUPE_ENABLED = os.getenv("AI_SEMANTIC_DEDUPE_ENABLED", "1") == "1"
AI_SEMANTIC_DEDUPE_MAX_ITEMS = int(os.getenv("AI_SEMANTIC_DEDUPE_MAX_ITEMS", "50"))
AI_SEMANTIC_DEDUPE_THRESHOLD = float(os.getenv("AI_SEMANTIC_DEDUPE_THRESHOLD", "0.88"))
ARTICLE_FETCH_ENABLED = os.getenv("ARTICLE_FETCH_ENABLED", "1") == "1"
ARTICLE_FETCH_MAX_ITEMS = int(os.getenv("ARTICLE_FETCH_MAX_ITEMS", "20"))
ARTICLE_FETCH_MIN_CHARS = int(os.getenv("ARTICLE_FETCH_MIN_CHARS", "400"))
ARTICLE_FETCH_TIMEOUT_SEC = int(os.getenv("ARTICLE_FETCH_TIMEOUT_SEC", "6"))
