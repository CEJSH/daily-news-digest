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
        "dedupeKey": "정책-발표-정부",
        "clusterKey": "정책/정부",
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


def test_detect_stale_incident_with_old_event_date() -> None:
    base = em._parse_date_base("2026-02-11")
    text = "정부는 2023년 11월 29일 발생한 침해 사고 조사 결과를 발표했다."
    stale, event_date = em._detect_stale_incident(base, text)
    assert stale is True
    assert event_date is not None
    assert event_date.year == 2023


def test_collect_item_errors_flags_stale_incident() -> None:
    item = _base_item(
        date="2026-02-11",
        publishedAt="2026-02-10T00:00:00+00:00",
        title="쿠팡 개인정보 유출 조사 결과",
        summary=["정부가 조사 결과를 발표했다."],
    )
    full_text = "정부는 2023년 11월 29일 발생한 침해 사고 조사 결과를 발표했다."
    errors = em._collect_item_errors(item, full_text=full_text, summary_text=item["summary"][0])
    assert "ERROR: STALE_INCIDENT_ITEM" in errors


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


def test_export_downgrades_importance_without_impact_signals(monkeypatch, tmp_path) -> None:
    import datetime
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    
    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 5)
    monkeypatch.setattr(em, "METRICS_JSON", str(tmp_path / "metrics.json"))
    output_path = tmp_path / "digest.json"
    ai_payload = {
        "summary_lines": ["정부가 정책을 발표했습니다."],
        "why_important": "정책 변화가 영향을 미칩니다.",
        "importance_rationale": "근거: 정부가 정책을 발표했습니다.",
        "importance_score": 4,
        "impact_signals": [],
        "quality_label": "ok",
        "quality_reason": "",
    }
    item = {
        "title": "정부 정책 발표",
        "summary": "정부가 정책을 발표했다.",
        "summaryRaw": "정부가 정책을 발표했다.",
        "fullText": "정부가 정책을 발표했다. " * 10,
        "topic": "정책",
        "source": "source",
        "sourceRaw": "source",
        "link": "http://example.com",
        "publishedAtUtc": date_str,
        "updatedAtUtc": date_str,
        "impactSignals": [],
        "dedupeKey": "정부-정책-발표",
        "clusterKey": "정책/정부",
        "score": 3.0,
        "ai": ai_payload,
    }
    digest = em.export_daily_digest_json(
        [item],
        str(output_path),
        {"selection_criteria": "", "editor_note": "", "question": ""},
    )
    assert digest["items"][0]["importance"] == 2
    # impactSignals가 없으면 importance가 2로 다운그레이드됨
    # qualityReason은 AI 결과에 따라 다른 값이 설정될 수 있음


def test_export_prefers_resolved_url_for_source_url(monkeypatch, tmp_path) -> None:
    import datetime

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    monkeypatch.setattr(em, "MIN_TOP_ITEMS", 1)
    monkeypatch.setattr(em, "TOP_LIMIT", 5)
    monkeypatch.setattr(em, "METRICS_JSON", str(tmp_path / "metrics.json"))
    output_path = tmp_path / "digest.json"
    item = {
        "title": "정부 정책 발표",
        "summary": "정부가 정책을 발표했다.",
        "summaryRaw": "정부가 정책을 발표했다.",
        "fullText": "정부가 정책을 발표했다. 적용 범위와 시행 시점을 설명했다. " * 5,
        "topic": "정책",
        "source": "source",
        "sourceRaw": "source",
        "link": "https://news.google.com/rss/articles/example",
        "resolvedUrl": "https://example.com/news/real-article",
        "publishedAtUtc": date_str,
        "updatedAtUtc": date_str,
        "impactSignals": ["policy"],
        "dedupeKey": "정부-정책-발표",
        "clusterKey": "정책/정부",
        "score": 3.0,
        "ai": {
            "summary_lines": ["정부가 정책을 발표했다."],
            "why_important": "정책 변화가 영향을 미친다.",
            "importance_rationale": "근거: 정부가 정책을 발표했다.",
            "importance_score": 3,
            "impact_signals": ["policy"],
            "impact_signals_evidence": {"policy": "정부가 정책을 발표했다."},
            "quality_label": "ok",
            "quality_reason": "",
        },
    }
    digest = em.export_daily_digest_json(
        [item],
        str(output_path),
        {"selection_criteria": "", "editor_note": "", "question": ""},
    )
    assert digest["items"][0]["sourceUrl"] == "https://example.com/news/real-article"
