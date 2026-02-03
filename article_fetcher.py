import re

from bs4 import BeautifulSoup
import requests

_WS_RE = re.compile(r"\s+")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}


def _clean_text(text: str) -> str:
    # 공백/개행을 하나의 공백으로 정리하고, 앞뒤 공백을 제거해
    # 본문 추출 결과를 비교/결합하기 쉬운 형태로 만든다.
    return _WS_RE.sub(" ", (text or "").strip())


def _extract_main_text(html: str) -> str:
    # HTML에서 본문으로 보이는 텍스트를 휴리스틱으로 추출한다.
    # 1) 광고/스크립트/네비 등 비본문 요소 제거
    # 2) article/main/section/div 중 길이가 충분한 텍스트 후보 수집
    # 3) 가장 긴 후보를 본문으로 선택 (없으면 body 전체 텍스트 반환)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "form"]):
        tag.decompose()
    for tag in soup.find_all(["header", "nav", "footer", "aside"]):
        tag.decompose()

    candidates: list[tuple[int, str]] = []
    for tag_name in ["article", "main", "section", "div"]:
        for tag in soup.find_all(tag_name):
            text = _clean_text(tag.get_text(" "))
            if len(text) >= 200:
                candidates.append((len(text), text))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    body = soup.body or soup
    return _clean_text(body.get_text(" "))


def _extract_canonical_url(html: str) -> str:
    # canonical 또는 og:url 메타에서 원문 URL을 찾아 반환한다.
    # (구글 뉴스 등 중간 리다이렉트 페이지를 원문으로 교체하기 위함)
    soup = BeautifulSoup(html, "html.parser")
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return link.get("href")
    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        return og.get("content")
    return ""


def fetch_article_text(
    url: str,
    timeout_sec: int = 6,
    max_chars: int = 12000,
) -> tuple[str, str]:
    # 주어진 URL에서 기사 HTML을 받아 본문 텍스트와 최종 URL을 반환한다.
    # - 요청 실패 시 빈 문자열 2개 반환
    # - Google News 링크인 경우 canonical/og:url로 원문 재요청
    # - 본문은 길이 제한(max_chars)까지 잘라 반환
    try:
        resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout_sec, allow_redirects=True)
    except Exception:
        return "", ""

    html = resp.text or ""
    final_url = resp.url or url

    if "news.google.com" in final_url:
        canonical = _extract_canonical_url(html)
        if canonical and canonical != final_url:
            try:
                resp = requests.get(canonical, headers=_DEFAULT_HEADERS, timeout=timeout_sec, allow_redirects=True)
                html = resp.text or ""
                final_url = resp.url or canonical
            except Exception:
                pass

    text = _extract_main_text(html)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text, final_url
