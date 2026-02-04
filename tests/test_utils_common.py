from daily_news_digest.utils import clean_text, ensure_lines_1_to_3, trim_title_noise


def test_clean_text_strips_html_and_ws() -> None:
    assert clean_text("  hello&nbsp;<b>world</b>\n") == "hello world"


def test_trim_title_noise_removes_tail_source() -> None:
    assert trim_title_noise("Some title - Reuters") == "Some title"


def test_ensure_lines_1_to_3_produces_lines() -> None:
    assert ensure_lines_1_to_3([], "A. B. C.") == ["A.", "B.", "C."]
