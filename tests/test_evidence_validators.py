from __future__ import annotations

from daily_news_digest.export.validators.evidence import (
    policy_evidence_valid,
    sanctions_evidence_valid,
    earnings_evidence_valid,
)


def test_policy_strong_keyword_passes() -> None:
    # POLICY_STRONG_KEYWORDS 포함: 통과
    assert policy_evidence_valid("국회가 새 법안을 통과시켰습니다.") is True


def test_policy_korean_regulator_without_strong_keyword_passes() -> None:
    # 한국어 규제기관 단독 evidence: 정책으로 인정되어야 함 (C1 회귀 가드)
    # "금융위"는 IMPACT_SIGNALS_MAP["policy"]에 있지만 POLICY_STRONG_KEYWORDS에는 없음
    assert policy_evidence_valid("금융위가 새로운 정책을 검토하고 있다.") is True


def test_policy_gov_with_negotiation_only_rejected() -> None:
    # 정부 + 협상만 있는 evidence: 정책이 아닌 정치 활동으로 분류
    assert policy_evidence_valid("정부와 외교 당국이 협상 회담을 이어갔다.") is False


def test_policy_trade_only_rejected() -> None:
    # 관세/무역/협상만 있고 binding policy 없음: 거부
    assert policy_evidence_valid("미국과 중국이 관세 협상을 진행 중이다.") is False


def test_policy_no_policy_keyword_rejected() -> None:
    # 정책 키워드 자체가 없으면 거부
    assert policy_evidence_valid("기업이 새로운 제품을 출시했습니다.") is False


def test_policy_empty_text_rejected() -> None:
    assert policy_evidence_valid("") is False
    assert policy_evidence_valid(None) is False  # type: ignore[arg-type]


def test_sanctions_keyword_required() -> None:
    assert sanctions_evidence_valid("미국이 추가 제재를 부과했다.") is True
    assert sanctions_evidence_valid("기업이 신제품을 발표했다.") is False


def test_earnings_requires_metric_and_number() -> None:
    assert earnings_evidence_valid("매출이 전년 대비 12% 증가했다.") is True
    # 숫자 없는 metric은 거부
    assert earnings_evidence_valid("매출이 늘었다고 밝혔다.") is False
    # metric 없는 숫자는 거부
    assert earnings_evidence_valid("회의에 100명이 참석했다.") is False
