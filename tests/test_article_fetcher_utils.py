from daily_news_digest.scrapers.article_fetcher_utils import (
    is_probably_non_article_url,
    looks_like_article_text,
)


def test_is_probably_non_article_url_detects_homepage_root() -> None:
    assert is_probably_non_article_url("https://www.sedaily.com/") is True


def test_is_probably_non_article_url_allows_article_path() -> None:
    assert is_probably_non_article_url("https://www.sedaily.com/NewsView/2GU7XW7ABC") is False


def test_looks_like_article_text_rejects_listing_noise() -> None:
    listing_like = (
        "마켓시그널 - 증권일반 DB증권 실적 발표 - Finance "
        "정책·제도 관련 기사 모음 - 부동산일반 주요 뉴스 정리 "
        "경제 - 추가 헤드라인 요약… 또 다른 헤드라인… 마지막 헤드라인…"
    )
    assert looks_like_article_text(listing_like, min_chars=50) is False


def test_looks_like_article_text_accepts_normal_article_body() -> None:
    article_text = (
        "삼성전자는 대법원 판결에 따라 목표달성 장려금을 퇴직금 산정에 반영하기로 했다. "
        "이번 조치는 임금 체계가 유사한 기업 전반의 인건비 구조에도 영향을 줄 수 있다는 분석이 나온다. "
        "회사는 세부 적용 기준과 일정도 함께 공지했다."
    )
    assert looks_like_article_text(article_text, min_chars=50) is True
