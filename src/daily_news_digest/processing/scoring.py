from __future__ import annotations

import datetime
from typing import Any, Callable

from daily_news_digest.core.constants import SANCTIONS_KEYWORDS, TRADE_TARIFF_KEYWORDS
from daily_news_digest.processing.types import Item


def map_topic_to_category(topic: str) -> str:
    t = (topic or "").lower()
    if not t:
        return "글로벌"

    if "글로벌_빅테크" in t or "빅테크" in t:
        return "IT"
    if t.startswith("it") or "it" in t or "tech" in t:
        return "IT"
    if "ai" in t or "반도체" in t or "보안" in t or "저작권" in t or "데이터" in t:
        return "IT"

    if "글로벌_정세" in t or "정세" in t or "외교" in t:
        return "글로벌"

    if "국내" in t:
        return "경제"
    if "정책" in t or "규제" in t:
        return "경제"
    if "실적" in t or "가이던스" in t:
        return "경제"
    if "투자" in t or "ipo" in t or "m&a" in t or "ma" in t:
        return "경제"
    if "전력" in t or "인프라" in t or "에너지" in t:
        return "경제"
    if "경제" in t:
        return "경제"

    if "글로벌" in t or "global" in t:
        return "글로벌"
    return "글로벌"


class ItemFilterScorer:
    def __init__(
        self,
        *,
        impact_signals_map: dict[str, list[str]],
        long_impact_signals: set[str],
        emotional_drop_keywords: list[str],
        drop_categories: set[str],
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

        return signals

    def map_topic_to_category(self, topic: str) -> str:
        return map_topic_to_category(topic)

    def get_item_category(self, item: Item) -> str:
        ai_category = item.get("aiCategory") or ""
        topic = item.get("topic", "")
        if ai_category == "글로벌" and "국내" in (topic or ""):
            return "경제"
        if ai_category in {"IT", "경제", "글로벌"}:
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
            if any(s in self._long_impact_signals for s in impact_signals):
                return True
            return False
        return True

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
            return True
        if matched_to:
            return True
        if not impact_signals:
            return True
        if not self.passes_freshness(age_hours, impact_signals):
            return True
        if not self.passes_emotional_filter(category, text_all, impact_signals):
            return True
        return False
