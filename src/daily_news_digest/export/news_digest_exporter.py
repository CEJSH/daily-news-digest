from __future__ import annotations

import datetime
from daily_news_digest.core.config import (
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
from daily_news_digest.processing.pipeline import build_default_pipeline
from daily_news_digest.export.export_manager import export_daily_digest_json


def _log(message: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


def main() -> None:
    try:
        _log("프로그램 시작")
        pipeline = build_default_pipeline(logger=_log)
        grouped_items, top_items = pipeline.fetch_grouped_and_top(RSS_SOURCES, top_limit=TOP_LIMIT)

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
