from __future__ import annotations

from daily_news_digest.processing import ai_enricher as ai


def test_parse_json_handles_fenced_and_trailing_commas() -> None:
    raw = """
    ```json
    {"a": 1,}
    ```
    """
    parsed = ai._parse_json(raw)
    assert parsed == {"a": 1}


def test_normalize_dedupe_key_drops_numbers_and_short_tokens() -> None:
    raw = "삼성 2024 반도체 한국 IT"
    assert ai._normalize_dedupe_key(raw) == "삼성-반도체-한국"


def test_normalize_summary_lines_filters_bad_and_uses_fallback() -> None:
    title = "타이틀"
    lines = ["타이틀", "자세한 내용은", "짧다"]
    fallback = "첫 번째 문장은 충분히 길고 완전한 문장입니다. 두 번째 문장도 정보를 제공합니다."
    cleaned = ai._normalize_summary_lines(lines, title, fallback)
    assert cleaned == [
        "첫 번째 문장은 충분히 길고 완전한 문장입니다.",
        "두 번째 문장도 정보를 제공합니다.",
    ]


def test_apply_quality_guardrails_marks_low_info() -> None:
    label, reason, tags = ai._apply_quality_guardrails(
        quality_label="ok",
        quality_reason="",
        quality_tags=[],
        title="제목",
        summary_lines=["짧은 문장입니다."],
        why_important="영향이 있습니다.",
    )
    assert label == "low_quality"
    assert reason == "정보 부족"
    assert tags == []


def test_filter_impact_signal_objects_adds_sanctions_when_evidence_present() -> None:
    items = [
        {"label": "policy", "evidence": "관세 인상 발표"},
        {"label": "sanctions", "evidence": "제재 발표"},
        {"label": "security", "evidence": "드론 공격"},
    ]
    candidates = ["policy"]
    full_text = "정부가 관세 인상 발표를 했고 제재 발표도 있었다. 드론 공격도 보고됐다."
    labels, evidence_map = ai._filter_impact_signal_objects(items, candidates, full_text)
    assert labels == ["policy", "sanctions"]
    assert evidence_map["sanctions"] == "제재 발표"
