from daily_news_digest.core.constants import normalize_source_name
from daily_news_digest.processing.scoring import ItemFilterScorer, map_topic_to_category


def test_map_topic_to_category_it() -> None:
    assert map_topic_to_category("글로벌_빅테크") == "IT"
    assert map_topic_to_category("AI_저작권_데이터권리") == "IT"


def test_map_topic_to_category_global() -> None:
    assert map_topic_to_category("글로벌_정세") == "글로벌"


def test_map_topic_to_category_economy() -> None:
    assert map_topic_to_category("실적_가이던스") == "경제"


def test_source_weight_case_insensitive() -> None:
    scorer = ItemFilterScorer(
        impact_signals_map={},
        long_impact_signals=set(),
        emotional_drop_keywords=[],
        drop_categories=set(),
        source_tier_a={"Reuters"},
        source_tier_b={"TechCrunch"},
        source_weight_enabled=True,
        source_weight_factor=0.6,
        top_source_allowlist=set(),
        top_source_allowlist_enabled=False,
        top_fresh_max_hours=72,
        top_fresh_except_signals=set(),
        top_fresh_except_max_hours=168,
        top_require_published=False,
    )
    assert scorer.source_weight("reuters") == 3.0
    assert scorer.source_weight("TECHCRUNCH") == 1.5


def test_source_weight_normalizes_korean_suffixes() -> None:
    scorer = ItemFilterScorer(
        impact_signals_map={},
        long_impact_signals=set(),
        emotional_drop_keywords=[],
        drop_categories=set(),
        source_tier_a={"연합뉴스"},
        source_tier_b=set(),
        source_weight_enabled=True,
        source_weight_factor=0.6,
        top_source_allowlist=set(),
        top_source_allowlist_enabled=False,
        top_fresh_max_hours=72,
        top_fresh_except_signals=set(),
        top_fresh_except_max_hours=168,
        top_require_published=False,
    )
    assert scorer.source_weight("연합뉴스 한민족센터") == 3.0


def test_normalize_source_name_removes_tv_suffix() -> None:
    assert normalize_source_name("매일경제TV") == "매일경제"
