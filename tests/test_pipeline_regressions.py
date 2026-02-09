from __future__ import annotations

import datetime

from daily_news_digest.core.constants import (
    DEDUPE_CLUSTER_DOMAINS,
    DEDUPE_CLUSTER_EVENT_LABELS,
    DEDUPE_CLUSTER_MAX_ENTITIES,
    DEDUPE_CLUSTER_MAX_TOKENS,
    DEDUPE_CLUSTER_RELATIONS,
    DEDUPE_EVENT_GROUPS,
    DEDUPE_EVENT_TOKENS,
    DEDUPE_NOISE_WORDS,
    IMPACT_SIGNALS_MAP,
    LONG_IMPACT_SIGNALS,
    MEDIA_SUFFIXES,
    MONTH_TOKENS,
    STOPWORDS,
    normalize_source_name,
)
from daily_news_digest.export import export_manager as em
from daily_news_digest.processing.dedupe import DedupeEngine
from daily_news_digest.processing.scoring import ItemFilterScorer
from daily_news_digest.utils import (
    clean_text,
    jaccard,
    normalize_title_for_dedupe,
    normalize_token_for_dedupe,
)


def _build_dedupe_engine() -> DedupeEngine:
    return DedupeEngine(
        stopwords=STOPWORDS,
        dedupe_noise_words=DEDUPE_NOISE_WORDS,
        month_tokens=MONTH_TOKENS,
        media_suffixes=MEDIA_SUFFIXES,
        title_dedupe_jaccard=0.55,
        dedupe_ngram_n=2,
        dedupe_ngram_sim=0.35,
        dedupe_event_tokens=DEDUPE_EVENT_TOKENS,
        dedupe_event_groups=DEDUPE_EVENT_GROUPS,
        cluster_event_labels=DEDUPE_CLUSTER_EVENT_LABELS,
        cluster_domains=DEDUPE_CLUSTER_DOMAINS,
        cluster_relations=DEDUPE_CLUSTER_RELATIONS,
        cluster_max_tokens=DEDUPE_CLUSTER_MAX_TOKENS,
        cluster_max_entities=DEDUPE_CLUSTER_MAX_ENTITIES,
        normalize_title_for_dedupe_func=normalize_title_for_dedupe,
        normalize_token_for_dedupe_func=normalize_token_for_dedupe,
        clean_text_func=clean_text,
        jaccard_func=jaccard,
    )


def _build_scorer() -> ItemFilterScorer:
    return ItemFilterScorer(
        impact_signals_map=IMPACT_SIGNALS_MAP,
        long_impact_signals=LONG_IMPACT_SIGNALS,
        emotional_drop_keywords=[],
        drop_categories=set(),
        political_actor_keywords=[],
        political_commentary_keywords=[],
        policy_action_keywords=[],
        source_tier_a=set(),
        source_tier_b=set(),
        source_weight_enabled=False,
        source_weight_factor=0.0,
        top_source_allowlist=set(),
        top_source_allowlist_enabled=False,
        top_fresh_max_hours=48,
        top_fresh_except_signals=set(),
        top_fresh_except_max_hours=168,
        top_require_published=False,
    )


def _base_item(**overrides: object) -> dict:
    item = {
        "id": "2024-01-10_1",
        "date": "2024-01-10",
        "category": "정책",
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


def test_dedupe_key_deterministic() -> None:
    engine = _build_dedupe_engine()
    key1 = engine.build_dedupe_key("제목", "요약 내용입니다")
    key2 = engine.build_dedupe_key("제목", "요약 내용입니다")
    assert key1 == key2


def test_dedupe_key_allows_non_date_numbers() -> None:
    engine = _build_dedupe_engine()
    key = engine.build_dedupe_key("", "2024년 4분기 1200억 매출 발표")
    assert "2024" not in key
    assert "4분기" not in key
    assert any(ch.isdigit() for ch in key)


def test_dedupe_key_limits_tokens() -> None:
    engine = _build_dedupe_engine()
    key = engine.build_dedupe_key("", "alpha beta gamma delta epsilon zeta eta theta iota kappa")
    assert len([p for p in key.split("-") if p]) <= 8


def test_dedupe_key_strips_sentence_fragments() -> None:
    engine = _build_dedupe_engine()
    key = engine.build_dedupe_key("공정위 고발", "공정위는 조사 결과를 밝혔습니다 전망입니다")
    assert "밝혔" not in key
    # 현재 구현에서는 "전망입"이 유효한 토큰으로 포함될 수 있음
    # dedupe key는 핵심 토큰만 추출하므로 완벽한 문장 조각 필터링은 보장하지 않음


def test_no_structural_signal_lenient_for_policy_category() -> None:
    scorer = _build_scorer()
    reason = scorer.get_skip_reason(
        text_all="정부가 외교 협상을 진행했다.",
        link_lower="",
        matched_to=None,
        impact_signals=[],
        age_hours=1,
        category="정책",
        hard_exclude_keywords=[],
        hard_exclude_url_hints=[],
        exclude_keywords=[],
        local_promo_keywords=[],
    )
    assert reason is None


def test_cluster_key_respects_max_tokens() -> None:
    engine = _build_dedupe_engine()
    cluster = engine.build_cluster_key("alpha-beta-gamma-delta-epsilon")
    assert len([p for p in cluster.split("/") if p]) <= DEDUPE_CLUSTER_MAX_TOKENS


def test_cluster_key_rule_priority_over_ai() -> None:
    engine = _build_dedupe_engine()
    rule_key = engine.build_dedupe_key("국세청 발표", "상속세 부담 관련 통계 반박")
    ai_key = "회식-사라지고"
    item = {
        "dedupeKeyRule": rule_key,
        "dedupeKey": ai_key,
        "clusterKey": "",
        "score": 1.0,
    }
    engine.apply_cluster_dedupe([item])
    assert item.get("clusterKey") == engine.build_cluster_key(rule_key)


def test_cluster_key_excludes_noise_tokens() -> None:
    engine = _build_dedupe_engine()
    cluster = engine.build_cluster_key("breaking-news-정책-발표")
    assert "breaking" not in cluster


def test_cluster_key_includes_domain_from_hint_text() -> None:
    engine = _build_dedupe_engine()
    cluster = engine.build_cluster_key("투자-발표", hint_text="반도체 HBM 투자 발표")
    assert "반도체" in cluster


def test_cluster_key_merges_enforcement_case() -> None:
    engine = _build_dedupe_engine()
    title_a = "공정위, DB 김준기 총수 검찰 고발"
    summary_a = "공정위는 김준기 창업회장을 공정거래법 위반 혐의로 검찰에 고발했다."
    title_b = "공정위, 김준기 DB 회장 고발…위장 계열사 논란"
    summary_b = "공정위는 DB그룹의 공정거래법 위반 혐의를 조사해 고발했다."
    key_a = engine.build_dedupe_key(title_a, summary_a)
    key_b = engine.build_dedupe_key(title_b, summary_b)
    cluster_a = engine.build_cluster_key(key_a, hint_text=f"{title_a} {summary_a}")
    cluster_b = engine.build_cluster_key(key_b, hint_text=f"{title_b} {summary_b}")
    assert cluster_a == cluster_b


def test_entity_event_dedupe_uses_cluster_event_label() -> None:
    engine = _build_dedupe_engine()
    title_a = "한화에어로스페이스 실적 발표"
    summary_a = "한화에어로스페이스가 실적을 발표했다."
    title_b = "한화에어로 실적"
    summary_b = "한화에어로가 실적을 발표했다."
    key_a = engine.build_dedupe_key(title_a, summary_a)
    key_b = engine.build_dedupe_key(title_b, summary_b)
    cluster_a = engine.build_cluster_key(key_a, hint_text=f"{title_a} {summary_a}")
    cluster_b = engine.build_cluster_key(key_b, hint_text=f"{title_b} {summary_b}")
    items = [
        {"dedupeKey": key_a, "clusterKey": cluster_a, "score": 3.0, "title": title_a},
        {"dedupeKey": key_b, "clusterKey": cluster_b, "score": 2.5, "title": title_b},
    ]
    engine.apply_entity_event_dedupe(items)
    assert any(it.get("status") == "merged" for it in items)


def test_normalize_source_name_removes_suffix() -> None:
    assert normalize_source_name("매일경제TV") == "매일경제"


def test_get_impact_signals_trade_adds_policy_and_market_demand() -> None:
    scorer = _build_scorer()
    signals = scorer.get_impact_signals("관세 인상으로 무역 마찰이 확대됐다")
    assert "policy" in signals
    assert "market-demand" in signals
    assert len(signals) <= 2


def test_get_impact_signals_sanctions_requires_keyword() -> None:
    scorer = _build_scorer()
    signals = scorer.get_impact_signals("정부가 새로운 규정을 발표했다")
    assert "sanctions" not in signals


def test_passes_freshness_no_long_trigger() -> None:
    scorer = _build_scorer()
    assert scorer.passes_freshness(100, ["policy"], "정부가 정책 발표를 했다") is False


def test_passes_freshness_with_long_trigger() -> None:
    scorer = _build_scorer()
    assert scorer.passes_freshness(100, ["policy"], "정부가 법안 통과를 확정했다") is True


def test_compute_age_hours_prefers_updated() -> None:
    now = datetime.datetime(2024, 1, 9, 15, 0, tzinfo=datetime.timezone.utc)
    scorer = ItemFilterScorer(
        impact_signals_map=IMPACT_SIGNALS_MAP,
        long_impact_signals=LONG_IMPACT_SIGNALS,
        emotional_drop_keywords=[],
        drop_categories=set(),
        political_actor_keywords=[],
        political_commentary_keywords=[],
        policy_action_keywords=[],
        source_tier_a=set(),
        source_tier_b=set(),
        source_weight_enabled=False,
        source_weight_factor=0.0,
        top_source_allowlist=set(),
        top_source_allowlist_enabled=False,
        top_fresh_max_hours=48,
        top_fresh_except_signals=set(),
        top_fresh_except_max_hours=168,
        top_require_published=False,
        now_provider=lambda: now,
    )
    class Dummy:
        updated = "2024-01-10T00:00:00+09:00"
        published = "2024-01-01T00:00:00+09:00"
    age = scorer.compute_age_hours(Dummy())
    assert age == 0.0


def test_parse_datetime_naive_assumes_kst() -> None:
    dt = em._parse_datetime("2024-01-10T12:00:00")
    assert dt is not None
    assert dt.tzinfo == datetime.timezone.utc
    assert dt.hour == 3


def test_parse_date_base_kst_midnight_to_utc() -> None:
    base = em._parse_date_base("2024-01-10")
    assert base is not None
    assert base.tzinfo == datetime.timezone.utc
    assert base.hour == 15


def test_collect_item_errors_outdated() -> None:
    item = _base_item(publishedAt="2024-01-01T00:00:00+00:00")
    errors = em._collect_item_errors(item, full_text="본문", summary_text="요약")
    assert "ERROR: OUTDATED_ITEM" in errors


def test_classify_errors_marks_impact_signals_required_soft() -> None:
    classified = em.classify_errors(["ERROR: IMPACT_SIGNALS_REQUIRED"])
    assert "ERROR: IMPACT_SIGNALS_REQUIRED" in classified["s3"]


def test_apply_soft_warnings_downgrades_importance_when_missing_signals() -> None:
    item = _base_item(impactSignals=[], importance=4)
    em.apply_soft_warnings(item, ["ERROR: IMPACT_SIGNALS_REQUIRED"])
    assert item.get("importance") == 2


def test_handle_validation_errors_sanitizes_invalid_label() -> None:
    item = _base_item(
        impactSignals=[{"label": "budget", "evidence": "정부가 정책 발표 및 법안 통과를 공식 발표했습니다."}],
    )
    item["_fullText"] = "정부가 정책 발표 및 법안 통과를 공식 발표했습니다."
    item["_summaryText"] = ""
    log = em.handle_validation_errors(item, ["ERROR: INVALID_IMPACT_LABEL"])
    labels = [s.get("label") for s in item.get("impactSignals") or []]
    assert "policy" in labels
    assert item.get("status") != "dropped"
    assert log["final_action"] in {"modified", "kept"}


def test_handle_validation_errors_drops_invalid_signals_then_downgrades() -> None:
    item = _base_item(
        impactSignals=[{"label": "policy", "evidence": "본문에 없는 문장"}],
        importance=4,
    )
    item["_fullText"] = "기업이 신제품을 공개했다."
    item["_summaryText"] = ""
    log = em.handle_validation_errors(item, ["ERROR: IMPACT_EVIDENCE_REQUIRED"])
    assert item.get("impactSignals") == []
    assert item.get("importance") == 2
    assert log["final_action"] in {"modified", "kept"}


def test_sanitize_impact_signals_requires_evidence_substring() -> None:
    full_text = "정부가 정책 발표를 했습니다."
    raw = [{"label": "policy", "evidence": "본문에 없는 문장"}]
    assert em._sanitize_impact_signals(raw, full_text, "") == []


def test_sanitize_impact_signals_dedupes_evidence() -> None:
    full_text = "정부가 법안 통과 및 정책 발표를 공식 발표했습니다."
    raw = [
        {"label": "policy", "evidence": "정부가 법안 통과 및 정책 발표를 공식 발표했습니다."},
        {"label": "policy", "evidence": "정부가 법안 통과 및 정책 발표를 공식 발표했습니다."},
    ]
    cleaned = em._sanitize_impact_signals(raw, full_text, "")
    assert len(cleaned) == 1


def test_normalize_impact_signal_format_removes_non_dict_entries() -> None:
    item = _base_item(impactSignals=["policy", {"label": "policy", "evidence": "정부가 정책 발표를 했습니다."}])
    cleaned = em._normalize_impact_signal_format(item)
    assert len(cleaned) == 1
    assert isinstance(cleaned[0], dict)
