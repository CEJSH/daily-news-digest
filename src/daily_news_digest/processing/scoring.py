from __future__ import annotations

import datetime
from typing import Any, Callable

from daily_news_digest.core.constants import SANCTIONS_KEYWORDS, TRADE_TARIFF_KEYWORDS
from daily_news_digest.processing.types import Item


def map_topic_to_category(topic: str) -> str:
    t = (topic or "").lower()
    if not t:
        return "국제"

    if "글로벌_빅테크" in t or "빅테크" in t:
        return "기술"
    if t.startswith("it") or "it" in t or "tech" in t:
        return "기술"
    if "ai" in t or "보안" in t or "저작권" in t or "데이터" in t or "클라우드" in t:
        return "기술"

    if (
        "정책" in t
        or "규제" in t
        or "입법" in t
        or "법안" in t
        or "국회" in t
        or "시행령" in t
        or "가이드라인" in t
        or "외교" in t
        or "통상" in t
        or "관세" in t
        or "무역" in t
        or "협상" in t
        or "제재" in t
        or "수출통제" in t
    ):
        return "정책"

    if "에너지" in t or "전력" in t or "전력망" in t or "원전" in t or "천연가스" in t or "lng" in t:
        return "에너지"
    if (
        "모빌리티" in t
        or "자동차" in t
        or "전기차" in t
        or "ev" in t
        or "자율주행" in t
        or "항공" in t
        or "우주" in t
        or "드론" in t
        or "철도" in t
        or "선박" in t
        or "해운" in t
    ):
        return "모빌리티"
    if "헬스" in t or "바이오" in t or "의료" in t or "제약" in t or "보건" in t:
        return "헬스"
    if "환경" in t or "기후" in t or "탄소" in t or "온실" in t or "배출" in t:
        return "환경"
    if (
        "금융" in t
        or "금리" in t
        or "환율" in t
        or "증시" in t
        or "채권" in t
        or "주가" in t
        or "ipo" in t
        or "m&a" in t
        or "투자" in t
        or "실적" in t
        or "가이던스" in t
    ):
        return "금융"
    if (
        "경제" in t
        or "경기" in t
        or "물가" in t
        or "gdp" in t
        or "pmi" in t
        or "고용" in t
        or "실업" in t
    ):
        return "경제"
    if (
        "산업" in t
        or "공급망" in t
        or "업계" in t
        or "제조" in t
        or "생산" in t
        or "공장" in t
        or "설비" in t
    ):
        return "산업"
    if "반도체" in t or "software" in t or "플랫폼" in t:
        return "기술"
    if "사회" in t or "노동" in t or "노조" in t or "인권" in t or "안전" in t:
        return "사회"
    if "라이프" in t or "생활" in t or "소비" in t or "유통" in t or "리테일" in t:
        return "라이프"

    if "국내" in t:
        return "경제"

    if "글로벌_정세" in t or "정세" in t or "글로벌" in t or "global" in t:
        return "국제"
    return "국제"


class ItemFilterScorer:
    def __init__(
        self,
        *,
        impact_signals_map: dict[str, list[str]],
        long_impact_signals: set[str],
        emotional_drop_keywords: list[str],
        drop_categories: set[str],
        political_actor_keywords: list[str],
        political_commentary_keywords: list[str],
        policy_action_keywords: list[str],
        source_tier_a: set[str],
        source_tier_b: set[str],
        source_weight_enabled: bool,
        source_weight_factor: float,
        top_source_allowlist: set[str],
        top_source_allowlist_enabled: bool,
        top_fresh_max_hours: int,
        top_fresh_except_signals: set[str],
        top_fresh_except_max_hours: int,
        top_require_published: bool,
        now_provider: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._impact_signals_map = impact_signals_map
        self._long_impact_signals = long_impact_signals
        self._emotional_drop_keywords = emotional_drop_keywords
        self._drop_categories = drop_categories
        self._political_actor_keywords = [k.lower() for k in political_actor_keywords]
        self._political_commentary_keywords = [k.lower() for k in political_commentary_keywords]
        self._policy_action_keywords = [k.lower() for k in policy_action_keywords]
        self._source_tier_a = source_tier_a
        self._source_tier_b = source_tier_b
        self._source_weight_enabled = source_weight_enabled
        self._source_weight_factor = source_weight_factor
        self._top_source_allowlist = {s for s in top_source_allowlist if s}
        self._top_source_allowlist_enabled = top_source_allowlist_enabled
        self._top_fresh_max_hours = top_fresh_max_hours
        self._top_fresh_except_signals = set(top_fresh_except_signals)
        self._top_fresh_except_max_hours = top_fresh_except_max_hours
        self._top_require_published = top_require_published
        self._now_provider = now_provider or (
            lambda: datetime.datetime.now(datetime.timezone.utc)
        )
        self._structural_context_signals = {
            "policy",
            "regulation-risk",
            "security",
            "infra",
            "infrastructure",
            "environment",
            "labor",
            "social-impact",
            "sanctions",
            "budget",
            "stats",
            "industry-trend",
        }
        self._structural_context_keywords = [
            "제도",
            "안전",
            "안전대책",
            "재발 방지",
            "재발방지",
            "관리 체계",
            "관리체계",
            "대책",
            "감독",
            "책임 규명",
            "책임규명",
            "규정",
            "규제",
            "지침",
            "표준",
            "기준",
            "safety",
            "regulation",
            "policy",
            "standard",
            "guideline",
        ]

    def get_impact_signals(self, text: str) -> list[str]:
        signals: list[str] = []
        text_lower = text.lower()
        for signal, keywords in self._impact_signals_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                signals.append(signal)

        # 관세/무역은 policy + market-demand 기본 신호로 간주
        has_trade = any(kw in text_lower for kw in TRADE_TARIFF_KEYWORDS)
        if has_trade:
            if "policy" not in signals:
                signals.append("policy")
            if "market-demand" not in signals:
                signals.append("market-demand")

        # 제재는 명시된 제재/수출통제 키워드가 있을 때만 허용
        has_sanctions = any(kw in text_lower for kw in SANCTIONS_KEYWORDS)
        if not has_sanctions and "sanctions" in signals:
            signals = [s for s in signals if s != "sanctions"]

        priority = [
            "policy",
            "earnings",
            "security",
            "capex",
            "investment",
            "market-demand",
            "sanctions",
            "regulation-risk",
            "industry-trend",
            "technology",
            "consumer-behavior",
            "labor",
            "health",
            "environment",
            "infrastructure",
            "social-impact",
            "budget",
            "stats",
            "infra",
        ]
        ordered = [s for s in priority if s in signals]
        return ordered[:2]

    def map_topic_to_category(self, topic: str) -> str:
        return map_topic_to_category(topic)

    def get_item_category(self, item: Item) -> str:
        ai_category = item.get("aiCategory") or ""
        topic = item.get("topic", "")
        if ai_category == "국제" and "국내" in (topic or ""):
            return self.map_topic_to_category(topic)
        if ai_category in {
            "경제",
            "산업",
            "기술",
            "금융",
            "정책",
            "국제",
            "사회",
            "라이프",
            "헬스",
            "환경",
            "에너지",
            "모빌리티",
        }:
            return ai_category
        return self.map_topic_to_category(topic)

    def source_weight(self, source_name: str) -> float:
        s = (source_name or "").strip().lower()
        if any(a.lower() in s for a in self._source_tier_a):
            return 3.0
        if any(b.lower() in s for b in self._source_tier_b):
            return 1.5
        return 0.3

    def source_weight_boost(self, source_name: str | None) -> float:
        if not self._source_weight_enabled:
            return 0.0
        if not source_name:
            return 0.0
        raw = self.source_weight(source_name)
        normalized = (raw - 0.3) / 2.7
        normalized = max(0.0, min(1.0, normalized))
        return normalized * self._source_weight_factor

    def compute_age_hours(self, entry: Any) -> float | None:
        published_parsed = getattr(entry, "published_parsed", None)
        if not published_parsed:
            return None
        published_dt = datetime.datetime(*published_parsed[:6], tzinfo=datetime.timezone.utc)
        now = self._now_provider()
        delta = now - published_dt
        return delta.total_seconds() / 3600.0

    def passes_freshness(self, age_hours: float | None, impact_signals: list[str]) -> bool:
        if age_hours is None:
            return True
        if age_hours > 168:
            return False
        if age_hours > 72 and not any(s in self._long_impact_signals for s in impact_signals):
            return False
        return True

    def passes_top_freshness(self, age_hours: float | None, impact_signals: list[str]) -> bool:
        if age_hours is None:
            return not self._top_require_published
        if age_hours <= self._top_fresh_max_hours:
            return True
        if any(s in self._top_fresh_except_signals for s in impact_signals):
            return age_hours <= self._top_fresh_except_max_hours
        return False

    def is_top_source_allowed(self, source_name: str | None) -> bool:
        if not self._top_source_allowlist_enabled:
            return True
        if not source_name:
            return False
        s = source_name.strip().lower()
        for allowed in self._top_source_allowlist:
            token = allowed.strip()
            if not token:
                continue
            if len(token) < 3:
                continue
            if token.lower() in s:
                return True
        return False

    def passes_emotional_filter(
        self,
        category: str,
        text_all: str,
        impact_signals: list[str],
    ) -> bool:
        if category in self._drop_categories:
            return False
        if any(k in text_all for k in self._emotional_drop_keywords):
            return self._has_structural_context(text_all, impact_signals)
        return True

    def _has_structural_context(self, text_all: str, impact_signals: list[str]) -> bool:
        if any(s in self._structural_context_signals for s in impact_signals):
            return True
        if any(k in text_all for k in self._policy_action_keywords):
            return True
        if any(k in text_all for k in self._structural_context_keywords):
            return True
        return False

    def score_entry(
        self,
        impact_signals: list[str],
        read_time_sec: int,
        source_name: str | None = None,
    ) -> float:
        score = 0.0
        if any(s in self._long_impact_signals for s in impact_signals):
            score += 3.0
        if any(s in ["capex", "infra", "security"] for s in impact_signals):
            score += 2.0
        if any(s in ["earnings", "market-demand"] for s in impact_signals):
            score += 1.0
        if read_time_sec <= 20:
            score += 0.5
        score += self.source_weight_boost(source_name)
        return score

    def is_eligible(self, item: Item) -> bool:
        if item.get("status") == "merged":
            return False
        return not item.get("dropReason")

    def should_skip_entry(
        self,
        *,
        text_all: str,
        link_lower: str,
        matched_to: str | None,
        impact_signals: list[str],
        age_hours: float | None,
        category: str,
        hard_exclude_keywords: list[str],
        hard_exclude_url_hints: list[str],
        exclude_keywords: list[str],
        local_promo_keywords: list[str],
    ) -> bool:
        if any(bad in text_all for bad in hard_exclude_keywords):
            return True
        if any(hint in link_lower for hint in hard_exclude_url_hints):
            return True
        if any(bad in text_all for bad in local_promo_keywords):
            return True
        if any(bad in text_all for bad in exclude_keywords):
            if not self._has_structural_context(text_all, impact_signals):
                return True
        if self._is_political_commentary(text_all):
            return True
        if matched_to:
            return True
        if not impact_signals and not self._has_structural_context(text_all, impact_signals):
            return True
        if not self.passes_freshness(age_hours, impact_signals):
            return True
        if not self.passes_emotional_filter(category, text_all, impact_signals):
            return True
        return False

    def _is_political_commentary(self, text_lower: str) -> bool:
        if not text_lower:
            return False
        if not any(k in text_lower for k in self._political_actor_keywords):
            return False
        if not any(k in text_lower for k in self._political_commentary_keywords):
            return False
        if any(k in text_lower for k in self._policy_action_keywords):
            return False
        return True
