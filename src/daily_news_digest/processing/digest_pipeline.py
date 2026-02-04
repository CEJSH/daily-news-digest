from __future__ import annotations

"""Compatibility facade for legacy imports.

Use `daily_news_digest.processing.pipeline` and related modules for new code.
"""

from daily_news_digest.processing.ai_service import AIEnrichmentService
from daily_news_digest.processing.constants import DEFAULT_TOP_MIX_TARGET
from daily_news_digest.processing.dedupe import DedupeEngine
from daily_news_digest.processing.parsing import EntryParser, SUMMARY_FALLBACK
from daily_news_digest.processing.pipeline import (
    DigestPipeline,
    build_default_ai_service,
    build_default_dedupe_engine,
    build_default_entry_parser,
    build_default_filter_scorer,
    build_default_pipeline,
)
from daily_news_digest.processing.scoring import ItemFilterScorer, map_topic_to_category
from daily_news_digest.processing.types import Item, LogFunc, ParseFunc

__all__ = [
    "AIEnrichmentService",
    "DEFAULT_TOP_MIX_TARGET",
    "DedupeEngine",
    "DigestPipeline",
    "EntryParser",
    "Item",
    "ItemFilterScorer",
    "LogFunc",
    "ParseFunc",
    "SUMMARY_FALLBACK",
    "build_default_ai_service",
    "build_default_dedupe_engine",
    "build_default_entry_parser",
    "build_default_filter_scorer",
    "build_default_pipeline",
    "map_topic_to_category",
]

