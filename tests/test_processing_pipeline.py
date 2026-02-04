from __future__ import annotations

import datetime

from daily_news_digest.processing.ai_service import AIEnrichmentService
from daily_news_digest.processing.dedupe import DedupeEngine
from daily_news_digest.processing.parsing import EntryParser
from daily_news_digest.processing.pipeline import DigestPipeline
from daily_news_digest.processing.scoring import ItemFilterScorer
from daily_news_digest.utils import clean_text, jaccard, normalize_title_for_dedupe, normalize_token_for_dedupe, trim_title_noise


class _Entry:
    def __init__(self, *, title: str, summary: str, published_parsed: tuple | None = None) -> None:
        self.title = title
        self.summary = summary
        if published_parsed is not None:
            self.published_parsed = published_parsed


class _Feed:
    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries


def _build_pipeline(feed: _Feed) -> DigestPipeline:
    scorer = ItemFilterScorer(
        impact_signals_map={"policy": ["policy"]},
        long_impact_signals=set(),
        emotional_drop_keywords=[],
        drop_categories=set(),
        source_tier_a=set(),
        source_tier_b=set(),
        source_weight_enabled=False,
        source_weight_factor=0.0,
        top_source_allowlist=set(),
        top_source_allowlist_enabled=False,
        top_fresh_max_hours=72,
        top_fresh_except_signals=set(),
        top_fresh_except_max_hours=168,
        top_require_published=False,
        now_provider=lambda: datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
    )
    dedupe = DedupeEngine(
        stopwords=set(),
        dedupe_noise_words=set(),
        month_tokens=set(),
        media_suffixes=(),
        title_dedupe_jaccard=0.6,
        dedupe_ngram_n=2,
        dedupe_ngram_sim=0.35,
        dedupe_event_tokens=set(),
        dedupe_event_groups={},
        normalize_title_for_dedupe_func=normalize_title_for_dedupe,
        normalize_token_for_dedupe_func=normalize_token_for_dedupe,
        clean_text_func=clean_text,
        jaccard_func=jaccard,
        is_eligible_func=scorer.is_eligible,
    )
    parser = EntryParser(clean_text_func=clean_text, trim_title_noise_func=trim_title_noise)
    ai_service = AIEnrichmentService(
        enrich_item_with_ai_func=None,
        get_embedding_func=None,
        fetch_article_text_func=None,
        estimate_read_time_func=lambda text: 10,
        score_entry_func=scorer.score_entry,
        get_item_category_func=scorer.get_item_category,
        is_eligible_func=scorer.is_eligible,
        logger=lambda _msg: None,
        ai_importance_enabled=False,
        ai_importance_max_items=0,
        ai_importance_weight=0.0,
        ai_quality_enabled=False,
        ai_semantic_dedupe_enabled=False,
        ai_semantic_dedupe_max_items=0,
        ai_semantic_dedupe_threshold=0.9,
        article_fetch_enabled=False,
        article_fetch_max_items=0,
        article_fetch_min_chars=0,
        article_fetch_timeout_sec=1,
    )
    return DigestPipeline(
        entry_parser=parser,
        filter_scorer=scorer,
        dedupe_engine=dedupe,
        ai_service=ai_service,
        feed_parser=lambda _url: feed,
        logger=lambda _msg: None,
        max_entries_per_feed=10,
        min_score=0.0,
        output_json="",
        dedupe_history_path="",
        dedupe_recent_days=0,
        top_mix_target={"IT": 1, "경제": 1, "글로벌": 1},
    )


def test_pipeline_handles_missing_link() -> None:
    entry = _Entry(title="Policy update", summary="policy impact is noted")
    feed = _Feed([entry])
    pipeline = _build_pipeline(feed)
    grouped, top_items = pipeline.fetch_grouped_and_top(
        sources=[{"topic": "IT", "url": "http://example.com/rss", "limit": 1}],
        top_limit=1,
    )
    assert grouped
    assert top_items
    assert "link" in top_items[0]
