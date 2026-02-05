from __future__ import annotations

HARD_EXCLUDE_KEYWORDS = [  # 보고서/행사/홍보성 등 즉시 제외 키워드
    "동향", "동향리포트", "리포트", "브리프", "백서", "자료집", "보고서", "연구보고서",
    "세미나", "웨비나", "컨퍼런스", "포럼", "행사", "모집", "신청", "접수", "보도자료",
    "홍보", "프로모션", "할인", "출시기념", "사설", "칼럼", "기고", "기자수첩",
    "whitepaper", "report", "brief", "webinar", "conference", "forum", "press release",
    "promotion", "apply now", "opinion", "editorial", "column", "commentary", "view",
    "must", "should"
]

HARD_EXCLUDE_URL_HINTS = [  # URL에 포함되면 제외하는 힌트 경로
    "/report", "/whitepaper", "/webinar", "/seminar", "/conference", "/event", "/download"
]

EXCLUDE_KEYWORDS = [  # 연예/스포츠/사건사고/감성성 기사 제외 키워드
    "연예", "스타", "걸그룹", "보이그룹", "아이돌", "배우", "가수", "예능", "드라마", "영화",
    "팬미팅", "컴백", "앨범", "뮤직비디오", "뮤비", "티저", "화보", "열애", "결별", "이혼",
    "결혼", "출산", "야구", "축구", "농구", "배구", "골프", "e스포츠", "K리그", "KBO",
    "프리미어리그", "챔피언스리그", "살해", "살인", "폭행", "성폭행", "강간", "납치", "사망",
    "시신", "징역", "맛집", "카페", "뷰맛집", "여행기", "관광지", "연휴", "날씨",
    "미세먼지", "교통통제", "경악", "발칵", "알고보니", "이유는", "근황", "포착", "망신",
    "누리꾼", "갑론을박", "결국", "정체", "충격", "헉", "소름", "이게 얼마", "대참사",
    "대박", "주의보", "레전드", "웃음", "웃겼", "눈물", "entertainment", "celebrity",
    "girl group", "boy group", "idol", "actor", "singer", "variety show", "drama", "movie",
    "fan meeting", "comeback", "album", "music video", "teaser", "photoshoot", "dating",
    "breakup", "divorce", "marriage", "childbirth", "baseball", "soccer", "basketball",
    "volleyball", "golf", "esports", "K League", "KBO", "Premier League",
    "Champions League", "murder", "killing", "assault", "sexual assault", "rape", "kidnapping",
    "death", "corpse", "police", "arrest", "detention", "trial", "prison sentence", "lawsuit",
    "restaurant", "cafe", "tour spot", "travel diary", "tourism", "holiday", "weather",
    "fine dust", "traffic control", "shock", "scandal", "caught on camera", "backlash",
    "controversy", "reason why", "latest update", "netizens", "argument", "eventually",
    "identity", "disaster", "huge", "warning", "legendary", "funny", "laughter", "tearful",
    "읍사무소", "면사무소", "마을회관", "체험 행사", "지역 소식", "전통시장", "지역주민",
    "마을 주민", "농촌 체험", "어촌 체험", "지역 축제", "군민", "공모 사업"
]

POLITICAL_ACTOR_KEYWORDS = [
    "국민의힘", "더불어민주당", "민주당", "여당", "야당", "의원", "대표", "원내대표",
    "대통령", "장관", "청와대", "국회", "정당",
]

POLITICAL_COMMENTARY_KEYWORDS = [
    "비판", "주장", "촉구", "지적", "발언", "주문", "강조", "남 탓", "책임",
    "공방", "논쟁", "압박", "경고",
]

POLICY_ACTION_KEYWORDS = [
    "법안", "개정", "시행령", "규정", "규제", "가이드라인", "인허가", "과징금",
    "의결", "통과", "발효", "시행", "유예", "합의", "처리",
]

SOURCE_TIER_A = {  # 신뢰도 상위 A 등급 소스(우선 선별)
    "Reuters", "Bloomberg", "Financial Times", "The Wall Street Journal", "WSJ",
    "연합뉴스", "한국경제", "매일경제", "서울경제"
}

SOURCE_TIER_B = {  # 신뢰도 B 등급 소스(보조 선별)
    "중앙일보", "동아일보", "한겨레", "경향신문", "머니투데이",
    "전자신문", "ZDNet Korea", "TechCrunch", "The Verge"
}

LOCAL_PROMO_KEYWORDS = [  # 지역 홍보·장터성 콘텐츠 제외 키워드
    "지역 특산품", "특산품", "홈쇼핑", "완판", "매진", "품절",
    "지역 축제", "지역 행사", "마을 축제", "농촌 체험", "어촌 체험", "체험 행사",
    "지역 소식", "지역 주민", "주민", "마을 주민", "읍사무소", "면사무소", "마을회관",
    "전통시장", "지역 상권", "지역경제 활성화", "관광객", "관광지", "맛집", "카페",
    "local festival", "local community", "town festival", "county fair", "home shopping",
    "sold out", "local residents", "community event"
]

DEDUPE_EVENT_TOKENS = {  # 이벤트 유형(투자·M&A·실적 등) 중복 판단 토큰
    "funding", "financing", "investment", "invests", "invest", "round", "series", "raise", "raised",
    "valuation", "capital", "funds",
    "acquisition", "acquire", "acquires", "merger", "m&a", "deal", "buyout", "stake", "takeover",
    "ipo", "listing", "listed", "offering",
    "earnings", "results", "guidance", "revenue", "profit", "loss",
    "sanction", "sanctions", "tariff", "tariffs", "export", "control",
    "policy", "regulation", "bill", "law", "guideline", "rule",
    "capex", "expansion", "build",
    "trade", "tariff", "tariffs", "negotiation", "negotiations", "talk", "talks", "summit", "meeting", "dialogue", "agreement",
    "투자", "투자유치", "자금조달", "펀딩", "라운드", "시리즈", "자금",
    "인수", "합병", "인수합병", "매각", "지분",
    "상장", "공모", "실적", "가이던스", "매출", "영업이익", "순이익", "실적발표",
    "제재", "관세", "수출통제", "규제", "법안", "정책", "가이드라인", "시행령",
    "증설", "설비", "투자계획",
    "무역", "협상", "회담", "정상회담", "협의", "대화", "합의", "협정",
}

DEDUPE_EVENT_GROUPS = {  # 이벤트 토큰을 카테고리로 묶어 중복 군집화
    "funding": {
        "funding", "financing", "investment", "invests", "invest", "raise", "raised",
        "round", "series", "valuation", "capital", "funds",
        "투자", "투자유치", "자금조달", "펀딩", "라운드", "시리즈", "자금",
    },
    "mna": {
        "acquisition", "acquire", "acquires", "merger", "m&a", "deal", "buyout", "stake", "takeover",
        "인수", "합병", "인수합병", "매각", "지분",
    },
    "ipo": {
        "ipo", "listing", "listed", "offering", "상장", "공모",
    },
    "earnings": {
        "earnings", "results", "guidance", "revenue", "profit", "loss",
        "실적", "가이던스", "매출", "영업이익", "순이익", "실적발표",
    },
    "sanctions": {
        "sanction", "sanctions", "export", "control", "제재", "수출통제",
    },
    "policy": {
        "policy", "regulation", "bill", "law", "guideline", "rule", "규제", "법안", "정책", "가이드라인", "시행령",
    },
    "capex": {
        "capex", "expansion", "build", "증설", "설비", "투자계획",
    },
    "trade_talks": {
        "trade", "tariff", "tariffs", "negotiation", "negotiations", "talk", "talks", "summit", "meeting", "dialogue",
        "agreement", "무역", "관세", "협상", "회담", "정상회담", "협의", "대화", "합의", "협정",
    },
}

# dedupe용 이슈 클러스터 상위 키(표준 라벨)
DEDUPE_CLUSTER_EVENT_LABELS = {
    "funding": "투자",
    "mna": "인수합병",
    "ipo": "상장",
    "earnings": "실적",
    "sanctions": "제재",
    "policy": "정책",
    "capex": "설비투자",
    "trade_talks": "관세",
}

# 클러스터 도메인 키워드(상위 주제)
DEDUPE_CLUSTER_DOMAINS = {
    "에너지": {
        "에너지", "전력", "전기", "전력망", "천연가스","natural gas" "lng", "원전", "원자력", "smr", "원자로",
        "energy", "power", "electricity", "nuclear",  "grid", "utility"
    },
    "반도체": {
        "반도체", "hbm", "메모리", "파운드리", "팹", "노광", "asml", "tsmc","euv", "수율", "계측", "미세공정", "나노", "2nm", "3nm", "gaa", "메트롤로지",   "패키징", "첨단패키징", "hpc", "칩", "fabless", "eda"
    },
    "ai": {
        "ai", "인공지능", "llm", "모델", "생성ai", "생성형", "gpu",  "npu", "accelerator", "추론", "학습", "training", "inference", "파라미터", "프롬프트"
    },
    "배터리": {
        "배터리", "전기차", "ev", "2차전지", "리튬", "양극재", "음극재","셀", "팩", "원통형", "파우치", "lfp", "ncm", "전해질", "분리막", "실리콘음극", "리사이클", "recycling"
    },
    "로봇": {
        "로봇", "robot","휴머노이드", "자율주행로봇", "로보틱스", "humanoid", "robotics"
    },
    "통신": {
        "통신", "5g", "6g",  "네트워크",
    },
    "클라우드": {
        "클라우드", "cloud", "데이터센터", "datacenter", "datacentre", "aws", "azure", "gcp", "hyperscaler","colocation", "코로케이션", "서버팜",
    },
}

# 주요 국가/지역 관계 상위 키
DEDUPE_CLUSTER_RELATIONS = {
    "한미": {"한국", "미국"},
    "미중": {"미국", "중국"},
    "한중": {"한국", "중국"},
    "한일": {"한국", "일본"},
    "미일": {"미국", "일본"},
    "미대만": {"미국", "대만"},
    "중대만": {"중국", "대만"},
    "한EU": {"한국", "유럽", "europe", "eu"},
    "미EU": {"미국", "유럽", "europe", "eu"},
    "중EU": {"중국", "유럽", "europe", "eu"},
    "미러": {"미국", "러시아", "러"},
    "중러": {"중국", "러시아", "러"},
    "미인도": {"미국", "인도"},
    "일중": {"일본", "중국"},
    "한북": {"한국", "북한"},
    "이스라엘이란": {"이스라엘", "이란"},
    "미중동": {"미국", "중동"},
}

DEDUPE_CLUSTER_MAX_TOKENS = 4
DEDUPE_CLUSTER_MAX_ENTITIES = 2

STOPWORDS = {  # 제목/요약 토큰 정규화 시 제거되는 불용어
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "with",
    "is", "are", "must", "should", "how", "become", "show", "little"
}

SANCTIONS_KEYWORDS = {
    "sanction", "sanctions", "export control", "entity list", "embargo", "asset freeze",
    "수출통제", "블랙리스트", "자산 동결", "자산동결", "금수", "금수조치", "embargo",
}

TRADE_TARIFF_KEYWORDS = {
    "tariff", "tariffs", "trade", "trade war", "trade talks", "negotiation", "agreement",
    "관세", "무역", "무역전쟁", "협상", "협정",
}

IMPACT_SIGNALS_MAP = {  # 영향도 신호어(카테고리별) 매핑
    "policy": [
        "bill", "law", "amendment", "regulation", "rule", "policy", "guideline", "government",
        "parliament", "congress", "penalty", "fine", "license", "approval", "supervision",
        "tariff", "tariffs", "trade", "trade talks", "negotiation", "agreement",
        "법안", "개정", "시행령", "규정", "규제", "국회", "정부", "금융위", "공정위",
        "과징금", "인허가", "감독", "제재", "가이드라인", "관세", "무역", "협상", "협정",
    ],
    "earnings": [
        "earnings", "guidance", "consensus", "profit", "loss", "margin", "forecast", "outlook",
        "revenue", "disclosure", "quarter", "q1", "q2", "q3", "q4", "upgrade", "downgrade",
        "매출", "영업이익", "순이익", "실적", "컨센서스", "가이던스", "전망", "마진", "상향", "하향",
    ],
    "capex": [
        "capex", "expansion", "build", "construction", "plant", "factory", "line",
        "data center", "datacentre", "facility", "capacity", "utilization",
        "증설", "설비", "시설", "공장", "데이터센터", "건설", "라인", "캐팩스", "가동률", "가동",
    ],
    "investment": [
        "funding", "financing", "fundraising", "raise", "raised", "round", "series",
        "investment", "invests", "invest", "equity", "stake", "valuation", "capital",
        "투자", "지분투자", "전략적 투자", "투자유치", "펀딩", "라운드", "시리즈",
        "자금조달", "벤처", "vc", "밸류에이션", "기업가치", "투자협상", "투자 협상",
    ],
    "market-demand": [
        "sales", "demand", "deliveries", "shipments", "orders", "bookings", "inventory",
        "price increase", "price decrease", "pricing", "consumption", "consumer sentiment",
        "판매", "수요", "출하", "주문", "예약", "재고", "가격 상승", "가격 하락", "소비 둔화",
    ],
    "security": [
        "breach", "hack", "leak", "attack", "ransomware", "cve", "vulnerability",
        "terror", "shooting", "public safety", "civil rights", "national security",
        "침해", "해킹", "유출", "공격", "랜섬웨어", "취약점", "민권", "총격", "테러", "안보",
    ],
    "sanctions": [
        "sanction", "sanctions", "export control", "entity list", "embargo", "asset freeze",
        "수출통제", "블랙리스트", "자산 동결", "자산동결", "금수", "금수조치", "embargo",
    ],
    "budget": [
        "budget", "fiscal", "appropriation", "incentive", "subsidy",
        "예산", "재정", "지원금", "세제혜택",
    ],
    "stats": [
        "cpi", "ppi", "inflation", "gdp", "pmi", "unemployment", "jobs report", "payrolls",
        "retail sales", "industrial production", "trade balance", "macro data", "economic data",
        "통계", "지표", "물가", "소비자물가", "생산자물가", "gdp", "pmi", "실업률", "고용지표",
        "고용", "수출입", "무역수지", "소매판매", "산업생산", "경제지표",
    ],
    "infra": ["outage", "downtime", "disruption", "장애", "정전", "서비스 중단"],
}

DEDUPE_NOISE_WORDS = {  # 중복 판정에서 의미 적은 노이즈 단어
    "bold", "little", "recovery", "shock", "inside", "first", "new", "top", "best",
    "strategy", "how", "why", "what", "where", "when", "show", "showcase", "unveils",
    "exclusive", "breaking", "update", "latest", "years", "after", "cornerstone", "become",
    "reuters", "bloomberg", "ft", "wsj", "financial", "times", "wall", "street", "journal",
    "연합뉴스", "매일경제", "한국경제", "서울경제", "머니투데이", "중앙일보", "동아일보",
    "한겨레", "경향신문", "techcrunch", "verge"
}

EMOTIONAL_DROP_KEYWORDS = ["참사", "충격", "분노", "논란", "폭로"]  # 감정 유발성 키워드 드롭
DROP_CATEGORIES = {"사회", "사건", "연예"}  # 카테고리 기준 즉시 제외

MONTH_TOKENS = {  # 날짜 토큰 정규화/필터링용 월 문자열
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june",
    "jul", "july", "aug", "august", "sep", "sept", "september", "oct", "october", "nov",
    "november", "dec", "december"
}

LONG_IMPACT_SIGNALS = {"policy", "budget", "sanctions", "stats"}  # 장기 영향 신호 카테고리
MEDIA_SUFFIXES = ("일보", "신문", "뉴스", "방송", "미디어", "tv", "TV")  # 언론사명 접미사 추정
