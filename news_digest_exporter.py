import datetime
from typing import Any

from config import (
    AFFILIATE_AD_TEXT,
    AFFILIATE_LINK,
    EDITOR_NOTE,
    NEWSLETTER_TITLE,
    OUTPUT_JSON,
    QUESTION_OF_THE_DAY,
    RSS_SOURCES,
    SELECTION_CRITERIA,
    TOP_LIMIT,
)
from digest_pipeline import (
    build_default_ai_service,
    build_default_dedupe_engine,
    build_default_entry_parser,
    build_default_filter_scorer,
    build_default_pipeline,
)
from export_manager import export_daily_digest_json

# ==========================================
# 핵심 로직 함수 (하위 호환성 유지)
# ==========================================

Item = dict[str, Any]


def _log(message: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def _new_entry_parser():
    return build_default_entry_parser()


def _new_filter_scorer():
    return build_default_filter_scorer()


def _new_dedupe_engine():
    scorer = _new_filter_scorer()
    return build_default_dedupe_engine(is_eligible_func=scorer.is_eligible)


def _new_ai_service():
    scorer = _new_filter_scorer()
    return build_default_ai_service(
        logger=_log,
        score_entry_func=scorer.score_entry,
        is_eligible_func=scorer.is_eligible,
    )


def _new_pipeline():
    return build_default_pipeline(logger=_log)


def get_impact_signals(text: str) -> list[str]:
    return _new_filter_scorer().get_impact_signals(text)


def _tokenize_for_dedupe(text: str) -> list[str]:
    return _new_dedupe_engine().tokenize_for_dedupe(text)


def _is_korean_token(token: str) -> bool:
    return _new_dedupe_engine().is_korean_token(token)


def _is_noise_token(token: str) -> bool:
    return _new_dedupe_engine().is_noise_token(token)


def _valid_token_length(token: str) -> bool:
    return _new_dedupe_engine().valid_token_length(token)


def _strip_source_from_text(text: str, source_name: str) -> str:
    return _new_entry_parser().strip_source_from_text(text, source_name)


def _pick_analysis_text(full_text: str, summary_clean: str) -> str:
    return _new_entry_parser().pick_analysis_text(full_text, summary_clean)


def _extract_full_text(entry: Any) -> str:
    return _new_entry_parser().extract_full_text(entry)


def _extract_entry_texts(entry: Any, source_name: str) -> tuple[str, str, str, str, str, str, str]:
    return _new_entry_parser().parse_entry(entry, source_name)


def _dedupe_key_ngrams(key: str, n: int) -> set[str]:
    return _new_dedupe_engine().dedupe_key_ngrams(key, n)


def get_dedupe_key(title: str, summary: str) -> str:
    return _new_dedupe_engine().build_dedupe_key(title, summary)


def map_topic_to_category(topic: str) -> str:
    return _new_filter_scorer().map_topic_to_category(topic)


def _get_item_category(item: Item) -> str:
    return _new_filter_scorer().get_item_category(item)


def source_weight(source_name: str) -> float:
    return _new_filter_scorer().source_weight(source_name)


def _source_weight_boost(source_name: str | None) -> float:
    return _new_filter_scorer().source_weight_boost(source_name)


def _compute_age_hours(entry: Any) -> float | None:
    return _new_filter_scorer().compute_age_hours(entry)


def _passes_freshness(age_hours: float | None, impact_signals: list[str]) -> bool:
    return _new_filter_scorer().passes_freshness(age_hours, impact_signals)


def _passes_emotional_filter(category: str, text_all: str, impact_signals: list[str]) -> bool:
    return _new_filter_scorer().passes_emotional_filter(category, text_all, impact_signals)


def score_entry(impact_signals: list[str], read_time_sec: int, source_name: str | None = None) -> float:
    return _new_filter_scorer().score_entry(impact_signals, read_time_sec, source_name)


def _is_eligible(item: Item) -> bool:
    return _new_filter_scorer().is_eligible(item)


def pick_top_with_mix(all_items: list[Item], top_limit: int = 5) -> list[Item]:
    return _new_pipeline().pick_top_with_mix(all_items, top_limit)


def _apply_ai_importance(items: list[Item]) -> None:
    _new_ai_service().apply_ai_importance(items)


def _apply_semantic_dedupe(items: list[Item]) -> None:
    _new_ai_service().apply_semantic_dedupe(items)


def _apply_dedupe_key_similarity(items: list[Item]) -> None:
    _new_dedupe_engine().apply_dedupe_key_similarity(items)


def _find_existing_duplicate(
    tokens: set[str],
    dedupe_key: str,
    dedupe_ngrams: set[str],
    seen_title_tokens: list[tuple[set[str], Item]],
    seen_items_by_dedupe_key: dict[str, Item],
    seen_dedupe_ngrams: list[tuple[set[str], Item]],
) -> Item | None:
    return _new_dedupe_engine().find_existing_duplicate(
        tokens,
        dedupe_key,
        dedupe_ngrams,
        seen_title_tokens=seen_title_tokens,
        seen_items_by_dedupe_key=seen_items_by_dedupe_key,
        seen_dedupe_ngrams=seen_dedupe_ngrams,
    )


def _should_skip_entry(
    *,
    text_all: str,
    link_lower: str,
    matched_to: str | None,
    impact_signals: list[str],
    age_hours: float | None,
    category: str,
    hard_exclude_keywords: list[str],
    hard_exclude_url_hints: list[str],
    exclude_keywords: list[str],
) -> bool:
    return _new_filter_scorer().should_skip_entry(
        text_all=text_all,
        link_lower=link_lower,
        matched_to=matched_to,
        impact_signals=impact_signals,
        age_hours=age_hours,
        category=category,
        hard_exclude_keywords=hard_exclude_keywords,
        hard_exclude_url_hints=hard_exclude_url_hints,
        exclude_keywords=exclude_keywords,
    )


def _load_recent_dedupe_map(digest_path: str, history_path: str, days: int) -> dict[str, str]:
    return _new_dedupe_engine().load_recent_dedupe_map(digest_path, history_path, days)


def fetch_news_grouped_and_top(
    sources: list[dict[str, Any]],
    top_limit: int = 3,
) -> tuple[dict[str, list[Item]], list[Item]]:
    return _new_pipeline().fetch_grouped_and_top(sources, top_limit)


def main() -> None:
    try:
        _log("프로그램 시작")
        grouped_items, top_items = fetch_news_grouped_and_top(RSS_SOURCES, top_limit=TOP_LIMIT)

        config = {
            "newsletter_title": NEWSLETTER_TITLE,
            "ad_text": AFFILIATE_AD_TEXT,
            "ad_link": AFFILIATE_LINK,
            "selection_criteria": SELECTION_CRITERIA,
            "editor_note": EDITOR_NOTE,
            "question": QUESTION_OF_THE_DAY,
        }

        export_daily_digest_json(top_items, OUTPUT_JSON, config)
        _log(f"완료! {OUTPUT_JSON} 파일이 생성되었습니다.")

    except Exception as e:
        print("❌ 오류 발생:", e)


if __name__ == "__main__":
    main()
