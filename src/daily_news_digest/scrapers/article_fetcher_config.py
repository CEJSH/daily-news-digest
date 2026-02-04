from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return raw in {"1", "true", "True", "yes", "YES"}


DEFAULT_USER_AGENTS: Tuple[str, ...] = (
    # Chrome 121 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36",
    # Chrome 121 (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36",
    # Chrome 120 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    # Chrome 120 (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    # Safari 17 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
)


@dataclass(frozen=True)
class ArticleFetcherConfig:
    default_timeout_sec: int = _env_int("ARTICLE_FETCH_TIMEOUT_SEC", 6)
    max_chars: int = _env_int("ARTICLE_FETCH_MAX_CHARS", 12000)
    min_text_chars: int = _env_int("ARTICLE_FETCH_TEXT_MIN_CHARS", 80)
    candidate_limit: int = _env_int("ARTICLE_FETCH_CANDIDATE_LIMIT", 20)
    candidate_log_limit: int = _env_int("ARTICLE_FETCH_CANDIDATE_LOG_LIMIT", 50)
    max_candidate_depth: int = _env_int("ARTICLE_FETCH_MAX_CANDIDATE_DEPTH", 1)
    post_nav_wait_ms: int = _env_int("ARTICLE_FETCH_POST_NAV_WAIT_MS", 0)
    google_news_wait_ms: int = _env_int("ARTICLE_FETCH_GOOGLE_NEWS_WAIT_MS", 600)
    google_news_wait_for_selector: str = os.getenv(
        "ARTICLE_FETCH_GOOGLE_NEWS_WAIT_SELECTOR",
        "a[href]",
    )
    context_pool_size: int = _env_int("ARTICLE_FETCH_CONTEXT_POOL_SIZE", 2)
    parallel_fetch_max_workers: int = _env_int("ARTICLE_FETCH_PARALLEL_MAX_WORKERS", 4)
    browser_locale: str = os.getenv("ARTICLE_FETCH_BROWSER_LOCALE", "en-US")
    headless: bool = _env_bool("ARTICLE_FETCH_HEADLESS", True)
    log_candidates: bool = _env_bool("ARTICLE_FETCH_LOG_CANDIDATES", True)
    google_news_referer: str = "https://news.google.com/"
    block_resource_types: Tuple[str, ...] = ("image", "media", "font", "stylesheet")
    user_agents: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_USER_AGENTS)
