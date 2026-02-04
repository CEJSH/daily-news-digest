def test_package_imports() -> None:
    import daily_news_digest  # noqa: F401

    from daily_news_digest.core import config  # noqa: F401
    from daily_news_digest.export import export_manager  # noqa: F401

    try:
        import pytest
    except Exception:
        return
    pytest.importorskip("feedparser")
    from daily_news_digest.processing import pipeline  # noqa: F401


def test_default_data_paths() -> None:
    from daily_news_digest.core.config import DEDUPE_HISTORY_PATH, OUTPUT_JSON

    assert OUTPUT_JSON.endswith("/data/daily_digest.json") or OUTPUT_JSON.endswith("\\data\\daily_digest.json")
    assert DEDUPE_HISTORY_PATH.endswith("/data/dedupe_history.json") or DEDUPE_HISTORY_PATH.endswith(
        "\\data\\dedupe_history.json"
    )
