"""Evidence 검증 함수들."""
from __future__ import annotations

import re
from typing import Any

from daily_news_digest.core.constants import (
    IMPACT_SIGNALS_MAP,
    MARKET_DEMAND_EVIDENCE_KEYWORDS,
    SANCTIONS_EVIDENCE_KEYWORDS,
    SECURITY_EVIDENCE_KEYWORDS,
)
from daily_news_digest.utils import clean_text

from daily_news_digest.export.constants import (
    CAPEX_ACTION_KEYWORDS,
    CAPEX_PLAN_KEYWORDS,
    EARNINGS_METRIC_KEYWORDS,
    INFRA_KEYWORDS,
    POLICY_GOV_KEYWORDS,
    POLICY_NEGOTIATION_KEYWORDS,
    POLICY_STRONG_KEYWORDS,
    POLICY_TRADE_ONLY_KEYWORDS,
)


def has_number_token(text: str) -> bool:
    """텍스트에 숫자 토큰이 있는지 확인."""
    if not text:
        return False
    t = clean_text(text)
    if not t:
        return False
    if re.search(r"\d", t):
        return True
    return any(unit in t for unit in ["억", "조", "만", "%", "달러", "원", "billion", "million", "trillion", "usd", "$"])


def policy_evidence_valid(text: str) -> bool:
    """Policy evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    policy_keywords = [k.lower() for k in IMPACT_SIGNALS_MAP.get("policy", [])]
    if not any(k in t for k in policy_keywords):
        return False
    if any(k in t for k in POLICY_STRONG_KEYWORDS):
        return True
    if any(k in t for k in POLICY_GOV_KEYWORDS) and any(k in t for k in POLICY_NEGOTIATION_KEYWORDS):
        return False
    if any(k in t for k in POLICY_TRADE_ONLY_KEYWORDS):
        return False
    return False


def sanctions_evidence_valid(text: str) -> bool:
    """Sanctions evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in SANCTIONS_EVIDENCE_KEYWORDS)


def market_demand_evidence_valid(text: str) -> bool:
    """Market-demand evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in MARKET_DEMAND_EVIDENCE_KEYWORDS)


def earnings_evidence_valid(text: str) -> bool:
    """Earnings evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k in t for k in EARNINGS_METRIC_KEYWORDS) and has_number_token(t)


def capex_evidence_valid(text: str) -> bool:
    """Capex evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    has_action = any(k in t for k in CAPEX_ACTION_KEYWORDS)
    has_plan = any(k in t for k in CAPEX_PLAN_KEYWORDS) or has_number_token(t)
    return has_action and has_plan


def infra_evidence_valid(text: str) -> bool:
    """Infra evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k in t for k in INFRA_KEYWORDS)


def security_evidence_valid(text: str) -> bool:
    """Security evidence 검증."""
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in SECURITY_EVIDENCE_KEYWORDS)


def label_evidence_valid(label: str, evidence: str) -> bool:
    """라벨에 해당하는 evidence 검증을 수행."""
    if label == "policy":
        return policy_evidence_valid(evidence)
    if label == "sanctions":
        return sanctions_evidence_valid(evidence)
    if label == "market-demand":
        return market_demand_evidence_valid(evidence)
    if label == "earnings":
        return earnings_evidence_valid(evidence)
    if label == "capex":
        return capex_evidence_valid(evidence)
    if label == "infra":
        return infra_evidence_valid(evidence)
    if label == "security":
        return security_evidence_valid(evidence)
    return False


def evidence_keyword_hits(label: str, evidence: str) -> int:
    """Label별 키워드가 evidence에 포함된 횟수 계산."""
    t = clean_text(evidence or "").lower()
    if not t:
        return 0
    
    keywords = []
    if label == "sanctions":
        keywords = SANCTIONS_EVIDENCE_KEYWORDS
    elif label == "market-demand":
        keywords = MARKET_DEMAND_EVIDENCE_KEYWORDS
    elif label == "security":
        keywords = SECURITY_EVIDENCE_KEYWORDS
    elif label == "policy":
        keywords = IMPACT_SIGNALS_MAP.get("policy", [])
    elif label == "capex":
        keywords = CAPEX_ACTION_KEYWORDS
    elif label == "earnings":
        keywords = EARNINGS_METRIC_KEYWORDS
    elif label == "infra":
        keywords = INFRA_KEYWORDS
    
    return sum(1 for kw in keywords if kw.lower() in t)


def evidence_specificity_score(label: str, evidence: str) -> tuple[int, int, int]:
    """Evidence 구체성 점수 (길이, 숫자, 키워드)."""
    length_score = len(clean_text(evidence or ""))
    number_score = 1 if has_number_token(evidence or "") else 0
    keyword_score = evidence_keyword_hits(label, evidence)
    return (length_score, number_score, keyword_score)
