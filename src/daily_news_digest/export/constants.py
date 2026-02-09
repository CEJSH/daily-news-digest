"""export_manager 관련 상수 정의."""
from __future__ import annotations

import datetime
from daily_news_digest.core.constants import ALLOWED_IMPACT_SIGNALS

# Impact signal 관련 상수
ALLOWED_IMPACT_LABELS = set(ALLOWED_IMPACT_SIGNALS)
IMPACT_LABEL_PRIORITY = ["sanctions", "policy", "security", "capex", "infra", "earnings", "market-demand"]
IMPACT_LEVEL_SCORE = {"long": 4, "med": 3, "low": 2}

# 시간대
KST = datetime.timezone(datetime.timedelta(hours=9))

# 단순 사건 키워드 (낮은 중요도)
SIMPLE_INCIDENT_KEYWORDS = [
    "사고", "화재", "폭발", "추락", "충돌", "교통사고", "정전", "붕괴", "침몰",
    "사망", "부상", "실종", "희생", "피해",
    "범죄", "사건", "살인", "폭행", "강간", "납치", "강도", "테러",
    "체포", "구속", "수사", "기소", "재판", "판결", "징역",
    "부고", "별세", "추모", "장례",
    "인사", "임명", "선임", "취임", "사임", "퇴임", "승진",
    "accident", "crash", "fire", "explosion", "collapse", "sinking",
    "death", "killed", "injured", "missing", "victim",
    "crime", "murder", "assault", "rape", "kidnapping", "robbery", "terror",
    "arrest", "detention", "investigation", "indictment", "trial", "sentence",
    "obituary", "died", "appointed", "resigned", "ceo", "chairman",
]

# Policy 관련 키워드
POLICY_ACTION_KEYWORDS = [
    "법안", "법률", "규제", "행정명령", "법 개정", "법개정", "정부 요구", "정책 발표",
    "통과", "의결", "시행", "발효", "공포", "가이드라인", "지침",
    "관세", "비관세", "무역 장벽", "trade barrier", "tariff", "non-tariff",
    "policy announcement", "official policy",
]
POLICY_NEGOTIATION_KEYWORDS = ["협상", "협의", "협정", "회담", "대화", "negotiation", "talks", "summit", "dialogue"]
POLICY_GOV_KEYWORDS = ["정부", "외교", "국가", "당국", "diplomatic", "government", "state"]
POLICY_TRADE_ONLY_KEYWORDS = [
    "협상", "협의", "협정", "회담", "대화",
    "관세", "무역", "무역전쟁", "trade", "tariff", "trade talks", "negotiation", "agreement", "summit", "dialogue",
]
POLICY_STRONG_KEYWORDS = [
    "법안", "법률", "규제", "행정명령", "법 개정", "법개정", "정책 발표",
    "통과", "의결", "시행", "발효", "공포", "가이드라인", "지침", "인허가", "과징금", "감독",
    "policy announcement", "official policy", "regulation", "rule", "guideline", "law", "bill",
]

# Sanctions 관련 키워드
SANCTIONS_REQUIRED_KEYWORDS = [
    "제재", "제재 발표", "제재 부과", "제재 확대", "자산 동결", "자산동결", "거래 금지", "거래금지",
    "블랙리스트", "수출통제", "수출 금지", "2차 제재",
    "sanction", "sanctions", "asset freeze", "assets frozen", "export ban", "secondary sanctions",
    "entity list", "export control",
]

# Market 관련 키워드
MARKET_VARIABLE_KEYWORDS = [
    "가격", "유가", "환율", "금리", "주가",
    "수요", "주문", "판매", "재고", "출하", "생산", "생산량",
    "price", "oil", "exchange rate", "fx", "interest rate", "stock",
    "demand", "orders", "sales", "inventory", "shipments", "deliveries", "production", "output",
]
MARKET_CHANGE_KEYWORDS = [
    "상승", "하락", "급등", "급락", "증가", "감소", "확대", "축소", "줄", "늘",
    "rise", "fall", "surge", "plunge", "increase", "decrease", "drop", "decline", "gain", "slump",
]

# Earnings 관련 키워드
EARNINGS_METRIC_KEYWORDS = [
    "매출", "영업이익", "영업익", "순이익", "순손실", "실적",
    "revenue", "operating profit", "operating income", "net income", "net profit", "earnings", "ebit", "ebitda",
]

# Capex 관련 키워드
CAPEX_ACTION_KEYWORDS = [
    "설비투자", "투자", "투자 계획", "투자계획", "투자 발표",
    "증설", "라인", "공장", "데이터센터", "시설", "건설", "착공",
    "capex", "expansion", "build", "construction", "plant", "factory", "data center",
]
CAPEX_PLAN_KEYWORDS = [
    "계획", "발표", "착공", "건설", "설립", "확대", "증설", "추진", "예정",
    "plan", "announce", "start", "begin", "expand",
]

# Infra 관련 키워드
INFRA_KEYWORDS = [
    "장애", "정전", "서비스 중단", "중단", "복구", "전력망", "망 장애", "통신 장애",
    "outage", "downtime", "disruption", "service disruption", "power grid", "network outage",
]

# Security 관련 키워드
SECURITY_INCIDENT_KEYWORDS = [
    "무력", "충돌", "공격", "격추", "군사", "군사 행동", "미사일", "드론", "폭격", "교전", "전투",
    "침해", "해킹", "랜섬웨어", "유출", "취약점", "보안 패치", "보안취약점", "제로데이", "cve",
    "해협 봉쇄", "해협봉쇄", "유조선",
    "attack", "strike", "shoot down", "intercept", "missile", "drone", "military", "conflict", "clash",
    "breach", "hack", "ransomware", "vulnerability", "zero-day", "patch", "cve", "blockade", "tanker",
]

# Alignment 트리거
ALIGNMENT_TRIGGERS = [
    "policy",
    "sanctions",
    "capex",
    "earnings",
    "tariff",
    "제재",
    "법안",
    "실적",
    "투자",
    "증설",
]
