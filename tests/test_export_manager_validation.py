from __future__ import annotations

from daily_news_digest.export import export_manager as em


def _base_item(**overrides: object) -> dict:
    item = {
        "id": "2024-01-10_1",
        "date": "2024-01-10",
        "category": "IT",
        "title": "정부 정책 발표",
        "summary": ["정부가 산업 정책을 발표하며 적용 범위를 설명했습니다."],
        "whyImportant": "정책 변화가 산업 전반에 영향을 줍니다.",
        "importanceRationale": "근거: 정부가 산업 정책을 발표하며 적용 범위를 설명했습니다.",
        "impactSignals": [{"label": "policy", "evidence": "정부가 산업 정책을 발표하며 적용 범위를 설명했습니다."}],
        "dedupeKey": "policy-change",
        "sourceName": "source",
        "sourceUrl": "http://example.com",
        "publishedAt": "2024-01-10T00:00:00+09:00",
        "status": "kept",
        "importance": 3,
        "qualityLabel": "ok",
        "qualityReason": "정보성 기사",
    }
    item.update(overrides)
    return item


def _build_digest(items: list[dict]) -> dict:
    return {
        "date": "2024-01-10",
        "selectionCriteria": "",
        "editorNote": "",
        "question": "",
        "lastUpdatedAt": "",
        "items": items,
    }


def test_validate_digest_rejects_duplicate_dedupe_key(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item_a = _base_item(dedupeKey="same-key")
    item_b = _base_item(id="2024-01-10_2", dedupeKey="same-key")
    valid, error = em._validate_digest(_build_digest([item_a, item_b]))
    assert valid is False
    assert error == "ERROR: DUPLICATE_DEDUPE_KEY"


def test_validate_digest_rejects_outdated_item(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item = _base_item(publishedAt="2024-01-01T00:00:00+00:00")
    valid, error = em._validate_digest(_build_digest([item]))
    assert valid is False
    assert error == "ERROR: OUTDATED_ITEM"


def test_validate_digest_requires_impact_signals_for_high_importance(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item = _base_item(
        impactSignals=[],
        title="기업 실적 발표",
        summary=["기업이 실적을 발표했습니다."],
        importance=3,
    )
    valid, error = em._validate_digest(_build_digest([item]))
    assert valid is False
    assert error == "ERROR: IMPACT_SIGNALS_REQUIRED"


def test_validate_digest_rejects_invalid_policy_evidence(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item = _base_item(
        impactSignals=[{"label": "policy", "evidence": "기업은 새로운 제품을 공개하며 세부 사항을 설명했습니다."}],
    )
    valid, error = em._validate_digest(_build_digest([item]))
    assert valid is False
    assert error == "ERROR: INVALID_POLICY_LABEL"


def test_validate_digest_rejects_invalid_impact_label(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item = _base_item(
        impactSignals=[{"label": "budget", "evidence": "정부가 예산을 확정했습니다."}],
    )
    valid, error = em._validate_digest(_build_digest([item]))
    assert valid is False
    assert error == "ERROR: INVALID_IMPACT_LABEL"


def test_validate_digest_rejects_missing_evidence(monkeypatch) -> None:
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 10)
    item = _base_item(
        impactSignals=[{"label": "policy", "evidence": ""}],
    )
    valid, error = em._validate_digest(_build_digest([item]))
    assert valid is False
    assert error == "ERROR: IMPACT_EVIDENCE_REQUIRED"


def test_sanitize_impact_signals_drops_missing_evidence() -> None:
    full_text = "정부가 정책 발표를 했습니다."
    raw = [{"label": "policy", "evidence": "기사에 없는 문장"}]
    assert em._sanitize_impact_signals(raw, full_text, "") == []


def test_sanitize_impact_signals_requires_substring() -> None:
    full_text = "미국 국무부는 제재 대상과 관련한 세부 내용을 발표했다."
    raw = [{"label": "sanctions", "evidence": "이 문장은 본문에 존재하지 않는다는 점을 강조했습니다."}]
    assert em._sanitize_impact_signals(raw, full_text, "") == []


def test_sanitize_impact_signals_requires_sanctions_keywords() -> None:
    full_text = "미국 국무부는 관련 조치를 설명하며 추가 발표를 이어갔다."
    raw = [{"label": "sanctions", "evidence": "미국 국무부는 관련 조치를 설명하며 추가 발표를 이어갔다."}]
    assert em._sanitize_impact_signals(raw, full_text, "") == []


def test_sanitize_impact_signals_requires_market_demand_keywords() -> None:
    full_text = "기업이 신규 전략을 발표했으며 사업 방향을 재정비한다고 밝혔다."
    raw = [{"label": "market-demand", "evidence": "기업이 신규 전략을 발표했으며 사업 방향을 재정비한다고 밝혔다."}]
    assert em._sanitize_impact_signals(raw, full_text, "") == []


def test_long_trigger_required_for_upgrade() -> None:
    signals = [{"label": "policy", "evidence": "정부가 정책 발표를 했습니다."}]
    assert em._infer_importance_from_signals(signals) == 2


def test_is_title_like_summary_detects_repetition() -> None:
    assert em._is_title_like_summary("테스트 제목", ["테스트 제목"]) is True
    assert em._is_title_like_summary("테스트 제목", ["다른 내용입니다."]) is False


def test_pick_summary_source_skips_title_only() -> None:
    assert em._pick_summary_source("제목", "요약", "제목") == ""
    assert em._pick_summary_source("제목", "요약", "본문 요약") == "본문 요약"
