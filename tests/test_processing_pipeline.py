from __future__ import annotations

import datetime

from daily_news_digest.processing.ai_service import AIEnrichmentService
import daily_news_digest.processing.pipeline as pipeline_mod
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


def test_pick_top_with_mix_enforces_policy_cap_and_minimums() -> None:
    pipeline = _build_pipeline(_Feed([]))
    base = {
        "status": "kept",
        "dropReason": "",
        "ageHours": 1,
        "impactSignals": [],
        "summary": "요약",
        "fullText": "본문",
    }
    items = [
        {"title": "정책1", "score": 10, "aiCategory": "정책", "source": "A", **base},
        {"title": "정책2", "score": 9, "aiCategory": "정책", "source": "B", **base},
        {"title": "정책3", "score": 8, "aiCategory": "정책", "source": "C", **base},
        {"title": "정책4", "score": 7, "aiCategory": "정책", "source": "D", **base},
        {"title": "산업", "score": 6, "aiCategory": "산업", "source": "E", **base},
        {"title": "경제", "score": 5, "aiCategory": "경제", "source": "F", **base},
        {"title": "국제", "score": 4, "aiCategory": "국제", "source": "G", **base},
    ]
    picked = pipeline.pick_top_with_mix(items, top_limit=5)
    categories = {}
    for item in picked:
        categories[item.get("aiCategory") or ""] = categories.get(item.get("aiCategory") or "", 0) + 1
    assert categories.get("정책", 0) <= 2
    assert categories.get("산업", 0) >= 1
    assert categories.get("경제", 0) >= 1
    assert categories.get("국제", 0) >= 1


def test_allowlist_strict_fallback_when_short(monkeypatch) -> None:
    pipeline = _build_pipeline(_Feed([]))
    scorer = pipeline._filter_scorer
    scorer._top_source_allowlist_enabled = True
    scorer._top_source_allowlist = {"good"}
    monkeypatch.setattr(pipeline_mod, "TOP_SOURCE_ALLOWLIST_ENABLED", True)
    monkeypatch.setattr(pipeline_mod, "TOP_SOURCE_ALLOWLIST_STRICT", True)

    base = {
        "status": "kept",
        "dropReason": "",
        "ageHours": 1,
        "impactSignals": [],
        "summary": "요약",
        "fullText": "본문",
    }
    items = [
        {"title": "GOOD1", "score": 5, "aiCategory": "경제", "source": "good", "sourceRaw": "good", **base},
        {"title": "BAD1", "score": 4, "aiCategory": "경제", "source": "bad", "sourceRaw": "bad", **base},
        {"title": "BAD2", "score": 3, "aiCategory": "경제", "source": "bad2", "sourceRaw": "bad2", **base},
    ]
    picked = pipeline.pick_top_with_mix(items, top_limit=3)
    assert len(picked) == 3
    assert any(it.get("source") == "bad" for it in picked)
