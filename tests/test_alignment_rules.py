from __future__ import annotations

from daily_news_digest.export import export_manager as em


def _sample_item(title: str, summary: str, *, dedupe_key: str, cluster_key: str) -> dict:
    return {
        "id": "2024-01-10_1",
        "date": "2024-01-10",
        "category": "산업",
        "title": title,
        "summary": [summary],
        "whyImportant": "정보성 기사",
        "importanceRationale": "근거: 본문 요약",
        "impactSignals": [],
        "dedupeKey": dedupe_key,
        "clusterKey": cluster_key,
        "sourceName": "source",
        "sourceUrl": "http://example.com",
        "publishedAt": "2024-01-10T00:00:00+09:00",
        "status": "kept",
        "importance": 1,
        "qualityLabel": "ok",
        "qualityReason": "정보성 기사",
    }


def test_alignment_rules_with_10_samples() -> None:
    engine = em._get_dedupe_engine()
    samples = [
        ("HBM3 대량생산", "SK하이닉스가 HBM3 대량생산을 시작했다."),
        ("전력망 투자 확대", "정부가 전력망 투자 계획을 발표했다."),
        ("AI 칩 수출 통제", "미국이 AI 칩 수출 통제를 강화했다."),
        ("클라우드 데이터센터 증설", "AWS가 신규 데이터센터 증설 계획을 밝혔다."),
        ("배터리 공급망 이슈", "배터리 원자재 공급망 위험이 커졌다."),
        ("반도체 장비 수요 증가", "EUV 장비 수요가 증가하고 있다."),
        ("IPO 일정 발표", "기술 기업이 IPO 일정을 발표했다."),
        ("전기차 수요 둔화", "전기차 판매 둔화로 재고가 증가했다."),
        ("원전 안전 규정 개정", "원전 안전 규정이 개정됐다."),
        ("통신망 장애 복구", "통신망 장애가 복구되면서 서비스가 정상화됐다."),
    ]
    for title, summary in samples:
        dedupe_key = engine.build_dedupe_key(title, summary)
        cluster_key = engine.build_cluster_key(dedupe_key, hint_text=f"{title} {summary}")
        item = _sample_item(title, summary, dedupe_key=dedupe_key, cluster_key=cluster_key)
        errors = em._collect_item_errors(item, full_text=summary, summary_text=summary)
        assert "ERROR: DEDUPE_KEY_NOT_ALIGNED" not in errors
        assert "ERROR: CLUSTER_KEY_NOT_ALIGNED" not in errors


def test_hbm3_dedupe_key_regenerated_when_misaligned() -> None:
    title = "HBM3 대량생산"
    summary = "SK하이닉스가 HBM3 대량생산을 시작했다."
    item = _sample_item(
        title,
        summary,
        dedupe_key="회식-혼술-주류",
        cluster_key="회식/혼술",
    )
    item["_fullText"] = summary
    item["_summaryText"] = summary
    errors = em._collect_item_errors(item, full_text=summary, summary_text=summary)
    assert "ERROR: DEDUPE_KEY_NOT_ALIGNED" in errors
    log = em.handle_validation_errors(item, errors, source_item=None)
    assert all(tok not in item.get("dedupeKey", "") for tok in ["회식", "혼술", "주류"])
    assert log["final_action"] in {"modified", "kept"}


def test_missing_impact_signals_for_high_importance_trigger() -> None:
    item = _sample_item(
        "법안 통과 소식",
        "정부가 새로운 법안을 통과시켰다.",
        dedupe_key="법안-통과",
        cluster_key="정책/법안",
    )
    item["importance"] = 3
    item["_fullText"] = item["summary"][0]
    item["_summaryText"] = item["summary"][0]
    errors = em._collect_item_errors(item, full_text=item["_fullText"], summary_text=item["_summaryText"])
    assert "ERROR: IMPACT_SIGNALS_MISSING_FOR_HIGH_IMPORTANCE" in errors
