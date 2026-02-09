from daily_news_digest.core.constants import STOPWORDS
from daily_news_digest.utils import clean_text, ensure_lines_1_to_3, normalize_token_for_dedupe, trim_title_noise


def test_clean_text_strips_html_and_ws() -> None:
    assert clean_text("  hello&nbsp;<b>world</b>\n") == "hello world"


def test_trim_title_noise_removes_tail_source() -> None:
    assert trim_title_noise("Some title - Reuters") == "Some title"


def test_ensure_lines_1_to_3_produces_lines() -> None:
    assert ensure_lines_1_to_3([], "A. B. C.") == ["A.", "B.", "C."]


def test_normalize_token_aliases_hanhwa_aero() -> None:
    assert normalize_token_for_dedupe("한화에어로스페이스", STOPWORDS) == "한화에어로"
    assert normalize_token_for_dedupe("한화에어로", STOPWORDS) == "한화에어로"
