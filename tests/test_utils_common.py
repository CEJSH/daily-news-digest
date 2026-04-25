from daily_news_digest.core.constants import STOPWORDS
from daily_news_digest.utils import (
    clean_text,
    ensure_lines_1_to_3,
    normalize_token_for_dedupe,
    strip_summary_boilerplate,
    trim_title_noise,
)


def test_clean_text_strips_html_and_ws() -> None:
    assert clean_text("  hello&nbsp;<b>world</b>\n") == "hello world"


def test_trim_title_noise_removes_tail_source() -> None:
    assert trim_title_noise("Some title - Reuters") == "Some title"


def test_ensure_lines_1_to_3_produces_lines() -> None:
    assert ensure_lines_1_to_3([], "A. B. C.") == ["A.", "B.", "C."]


def test_normalize_token_aliases_hanhwa_aero() -> None:
    assert normalize_token_for_dedupe("한화에어로스페이스", STOPWORDS) == "한화에어로"
    assert normalize_token_for_dedupe("한화에어로", STOPWORDS) == "한화에어로"


def test_strip_summary_boilerplate_removes_korean_footer() -> None:
    raw = (
        "한국에서 발표된 새로운 AI 정책 발표문입니다. 핵심 내용은 다음과 같습니다.\n"
        "ⓒ 무단전재 및 재배포 금지\n"
        "주소 : 서울시 강남구 테헤란로 123\n"
        "대표전화 02-123-4567"
    )
    out = strip_summary_boilerplate(raw)
    assert "주소" not in out
    assert "대표전화" not in out
    assert "무단전재" not in out
    assert "한국에서 발표된" in out


def test_strip_summary_boilerplate_removes_english_copyright() -> None:
    raw = (
        "OpenAI announced a new model with extended reasoning capabilities.\n"
        "Copyright 2025 All Rights Reserved\n"
        "info@example.com"
    )
    out = strip_summary_boilerplate(raw)
    assert "Copyright" not in out
    assert "Rights Reserved" not in out
    assert "OpenAI announced" in out


def test_strip_summary_boilerplate_handles_inline_copyright() -> None:
    raw = "Reuters reports new tariff details. Copyright 2025 Reuters. All Rights Reserved"
    out = strip_summary_boilerplate(raw)
    assert "Reuters reports new tariff details" in out
    assert "All Rights Reserved" not in out


def test_strip_summary_boilerplate_returns_empty_for_empty_input() -> None:
    assert strip_summary_boilerplate("") == ""
    assert strip_summary_boilerplate(None) == ""  # type: ignore[arg-type]


def test_strip_summary_boilerplate_trims_trailing_separators() -> None:
    raw = "본문 핵심 내용입니다 ·"
    out = strip_summary_boilerplate(raw)
    assert out.endswith("내용입니다")
