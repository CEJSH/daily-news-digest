from __future__ import annotations

QUALITY_KEYWORDS = [  # 고품질/심층 기사 신호어 (가중치·선별에 사용)
    "분석", "해설", "전망", "심층", "진단", "전략", "패권", "패러다임", "변곡점", "구조",
    "재편", "지형", "모멘텀", "구조적", "생태계", "시나리오", "data", "in-depth",
    "diagnosis", "strategy", "paradigm", "inflection point", "structure", "reorganization",
    "ecosystem", "scenario"
]

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

STOPWORDS = {  # 제목/요약 토큰 정규화 시 제거되는 불용어
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "with",
    "is", "are", "must", "should", "how", "become", "show", "little"
}

SANCTIONS_KEYWORDS = {
    "sanction", "sanctions", "export control", "entity list", "embargo", "asset freeze",
    "제재", "수출통제", "블랙리스트", "금융제재", "자산동결",
}

TRADE_TARIFF_KEYWORDS = {
    "tariff", "tariffs", "trade", "trade war", "trade talks", "negotiation", "agreement",
    "관세", "무역", "무역전쟁", "협상", "협정",
}

IMPACT_SIGNALS_MAP = {  # 영향도 신호어(카테고리별) 매핑
    "policy": [
        "regulation", "rule", "policy", "bill", "law", "guideline", "government",
        "tariff", "tariffs", "trade", "trade talks", "negotiation", "agreement",
        "규제", "법안", "정책", "가이드라인", "정부", "국회", "관세", "무역", "협상", "협정",
    ],
    "budget": ["budget", "fiscal", "appropriation", "incentive", "subsidy", "예산", "재정", "지원금", "세제혜택"],
    "sanctions": [
        "sanction", "sanctions", "export control", "entity list", "embargo", "asset freeze",
        "제재", "수출통제", "블랙리스트", "금융제재", "자산동결",
    ],
    "capex": ["data center", "datacentre", "capex", "investment", "build", "expansion", "infrastructure", "facility", "데이터센터", "증설", "투자", "설비"],
    "earnings": ["earnings", "guidance", "profit", "loss", "revenue", "흑자", "적자", "실적", "가이던스", "매출", "영업이익"],
    "stats": [
        "cpi", "ppi", "inflation", "gdp", "pmi", "unemployment", "jobs report", "payrolls",
        "retail sales", "industrial production", "trade balance", "macro data", "economic data",
        "통계", "지표", "물가", "소비자물가", "생산자물가", "gdp", "pmi", "실업률", "고용지표",
        "고용", "수출입", "무역수지", "소매판매", "산업생산", "경제지표",
    ],
    "market-demand": [
        "registrations", "registration", "deliveries", "delivery", "sales", "demand", "shipments",
        "tariff", "tariffs", "trade", "관세", "무역",
        "등록", "판매", "수요",
    ],
    "security": ["breach", "exploit", "ransomware", "cve", "vulnerability", "침해", "해킹", "랜섬웨어", "취약점"],
    "infra": ["outage", "downtime", "disruption", "장애", "정전", "서비스 중단"]
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
