from daily_news_digest.utils import clean_text_ws, contains_binary, jaccard_tokens, sanitize_text


def test_clean_text_ws_collapses() -> None:
    assert clean_text_ws("  hello   world \n") == "hello world"


def test_contains_binary_detects_control() -> None:
    assert contains_binary("ok\x01bad") is True


def test_sanitize_text_strips_binary() -> None:
    assert sanitize_text("ok\x01bad") == "ok bad"


def test_jaccard_tokens_basic() -> None:
    assert jaccard_tokens("a b c", "b c d") == 2 / 4

