import random
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

_WS_RE = re.compile(r"\s+")

_UA_POOL = [
    # Chrome 121 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36",
    # Chrome 121 (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36",
    # Chrome 120 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    # Chrome 120 (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    # Safari 17 (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

def _pick_user_agent() -> str:
    return random.choice(_UA_POOL)


def _make_headers() -> dict[str, str]:
    return {
        "User-Agent": _pick_user_agent(),
        "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    }

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

try:
    from newspaper import Article  # type: ignore[import-not-found]
    from newspaper import Config as NewspaperConfig  # type: ignore[import-not-found]
except Exception:
    Article = None  # type: ignore[assignment]
    NewspaperConfig = None  # type: ignore[assignment]

try:
    import trafilatura  # type: ignore[import-not-found]
except Exception:
    trafilatura = None  # type: ignore[assignment]


# -----------------------------
# Public return types
# -----------------------------
@dataclass(frozen=True)
class FetchMeta:
    requested_url: str
    final_url: str
    status: int
    html_len: int
    extractor: str  # "trafilatura" | "newspaper" | "heuristic" | "none"
    notes: list[str]


@dataclass(frozen=True)
class FetchResult:
    text: str
    meta: FetchMeta


# -----------------------------
# Utilities
# -----------------------------
def _clean_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _safe_decode_response(resp: requests.Response) -> str:
    """
    왜: resp.text는 인코딩 추정이 틀리면 한글이 깨질 수 있음.
    해결: apparent_encoding으로 보정 후 text를 읽는다.
    """
    try:
        if not resp.encoding:
            resp.encoding = resp.apparent_encoding
        else:
            # 일부 사이트는 잘못된 encoding을 주기도 해서, apparent_encoding이 더 그럴듯하면 교체
            apparent = resp.apparent_encoding or ""
            if apparent and resp.encoding.lower() != apparent.lower():
                # 너무 공격적으로 바꾸면 오히려 깨질 수 있어, 실패 시 기존 유지 로직은 여기선 단순화.
                resp.encoding = apparent
    except Exception:
        pass
    return resp.text or ""

def _is_textual_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    return any(x in ct for x in ["text/", "application/json", "application/xml", "application/xhtml+xml"])


def _looks_like_png_bytes(content: bytes) -> bool:
    return content.startswith(_PNG_SIG)


def _response_body_ok(resp: requests.Response) -> tuple[bool, str, bytes]:
    raw = resp.content or b""
    ct = (resp.headers.get("Content-Type") or "").strip().lower()
    clen = (resp.headers.get("Content-Length") or "").strip()
    if _looks_like_png_bytes(raw):
        size_hint = clen or str(len(raw)) if raw else ""
        reason = "png_bytes_detected"
        if ct:
            reason += f":{ct}"
        if size_hint:
            reason += f":len={size_hint}"
        return False, reason, raw
    if not _is_textual_response(resp):
        size_hint = clen or str(len(raw)) if raw else ""
        reason = f"non_textual_content_type:{ct or 'missing'}"
        if size_hint:
            reason += f":len={size_hint}"
        return False, reason, raw
    return True, "", raw


def _is_google_news(url: str) -> bool:
    try:
        return urlparse(url).netloc.endswith("news.google.com")
    except Exception:
        return False

_MEDIA_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp4",
    ".webm",
    ".mp3",
    ".pdf",
)
_MEDIA_HOST_HINTS = (
    "gstatic.com",
    "googleusercontent.com",
)


def _is_probably_media_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if any(host.endswith(h) or h in host for h in _MEDIA_HOST_HINTS):
        return True
    if any(path.endswith(ext) for ext in _MEDIA_EXTENSIONS):
        return True
    if "gen_204" in path:
        return True
    return False


def _same_domain(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False


def _normalize_url(base: str, href: str) -> str:
    # 왜: Google News는 상대경로(./articles/...)가 많음 → 절대경로로 합쳐야 함
    return urljoin(base, href)


# -----------------------------
# Extract candidate external/original URLs
# -----------------------------
def _extract_canonical_url(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return _normalize_url(base_url, link.get("href"))

    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        return _normalize_url(base_url, og.get("content"))

    refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
    if refresh and refresh.get("content"):
        m = re.search(r"url=(.+)", refresh["content"], flags=re.IGNORECASE)
        if m:
            return _normalize_url(base_url, m.group(1).strip())

    return ""


def _extract_external_hrefs_from_google_news(html: str, base_url: str, limit: int = 20) -> list[str]:
    """
    왜:
    - Google News는 외부 원문 링크가 <a href="./articles/..."> 내부 페이지로 숨겨져 있거나,
      외부 링크가 있어도 상대경로/추적 링크가 섞여있음.
    - "첫 번째 외부 http 링크"만 집는 건 광고/추천 링크에 쉽게 걸림.

    전략:
    - a[href]를 다 수집 → 절대경로로 정규화
    - news.google.com 내부 링크(articles 등) + 외부 링크를 모두 후보로 반환
    - caller에서 "내부 링크면 한 번 더 열어서 외부 원문을 찾는" 방식으로 사용
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = _normalize_url(base_url, href)
        # 불필요한 앵커/자바스크립트 제거
        if abs_url.startswith("javascript:") or abs_url.startswith("#"):
            continue
        if _is_probably_media_url(abs_url):
            continue
        out.append(abs_url)
        if len(out) >= limit:
            break

    # 마지막 보험: 정규식으로 https URL도 추가
    if len(out) < limit:
        for u in re.findall(r"https?://[^\s\"'>]+", html):
            if _is_probably_media_url(u):
                continue
            out.append(u)
            if len(out) >= limit:
                break

    # 중복 제거(순서 유지)
    seen = set()
    uniq: list[str] = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def _pick_best_external_url(candidates: list[str], current_url: str) -> Optional[str]:
    """
    왜:
    - 후보 중 외부 원문을 고를 때, 단순 첫 번째는 위험(광고/추천/로그인 등)
    - 여기서는 "google news가 아닌 링크"를 우선.
    - 단, 외부 링크가 없다면 google news 내부 articles 링크를 반환해서 2차 추출을 유도.
    """
    if not candidates:
        return None

    # 1) news.google.com이 아닌 링크 우선
    for u in candidates:
        if _is_probably_media_url(u):
            continue
        if not _is_google_news(u):
            return u

    # 2) 전부 google news면, 현재 URL과 다른 것으로
    for u in candidates:
        if _is_probably_media_url(u):
            continue
        if u != current_url:
            return u

    return None


# -----------------------------
# Text extraction backends
# -----------------------------
def _extract_with_trafilatura(url: str, html: str) -> str:
    if trafilatura is None:
        return ""
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
        )
        return _clean_text(extracted or "")
    except Exception:
        return ""


def _extract_with_newspaper(url: str, html: str, timeout_sec: int) -> str:
    if Article is None:
        return ""
    try:
        config = None
        if NewspaperConfig is not None:
            config = NewspaperConfig()
            config.browser_user_agent = _DEFAULT_HEADERS.get("User-Agent", "")
            config.request_timeout = timeout_sec
            config.fetch_images = False

        article = Article(url, config=config) if config is not None else Article(url)
        try:
            article.download(input_html=html)
        except TypeError:
            article.download()
        article.parse()
        return _clean_text(getattr(article, "text", "") or "")
    except Exception:
        return ""


def _text_quality_score(tag) -> float:
    """
    왜:
    - "가장 긴 텍스트"는 댓글/추천/푸터에 자주 속음.
    - 링크 비율/문장부호/문장 길이 등을 가볍게 반영해 본문 후보를 더 잘 고름.

    점수는 휴리스틱이며, 운영에서 튜닝 포인트.
    """
    text = _clean_text(tag.get_text(" "))
    if not text:
        return -1.0

    length = len(text)

    # 링크 텍스트 비율(높으면 네비/추천일 가능성 ↑)
    link_text_len = 0
    for a in tag.find_all("a"):
        link_text_len += len(_clean_text(a.get_text(" ")))
    link_ratio = (link_text_len / max(1, length))

    # 문장부호 밀도(본문은 대체로 .?!, 한국어는 '다.' 같은 종결도 포함될 수 있음)
    punct = sum(text.count(c) for c in [".", "?", "!", "。", "！", "？"])
    punct_density = punct / max(1, length)

    # 너무 짧으면 후보 가치 낮음
    if length < 120:
        return -1.0

    # 스코어: 길이 가중 + 문장부호 약간 + 링크비율 페널티
    # 길이는 log로 완만하게(너무 긴 푸터/댓글이 과대평가되는 걸 완화)
    import math

    score = math.log(length)
    score += min(0.8, punct_density * 50)  # 과도한 영향 제한
    score -= link_ratio * 3.5

    # 과도하게 링크가 많으면 강력 페널티
    if link_ratio > 0.35:
        score -= 2.0

    return score


def _extract_main_text_heuristic(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 비본문 제거
    for tag in soup(["script", "style", "noscript", "iframe", "form"]):
        tag.decompose()
    for tag in soup.find_all(["header", "nav", "footer", "aside"]):
        tag.decompose()

    candidates = []

    # article/main 우선
    for tag_name in ["article", "main", "section", "div"]:
        for tag in soup.find_all(tag_name):
            score = _text_quality_score(tag)
            if score > 0:
                candidates.append((score, tag))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_tag = candidates[0][1]
        return _clean_text(best_tag.get_text(" "))

    # 최후: body 전체
    body = soup.body or soup
    return _clean_text(body.get_text(" "))


def _looks_like_article_text(text: str) -> bool:
    """
    왜:
    - 길이만으로 성공/실패 판단하면 속보/에러페이지 둘 다 오판 가능.
    - 최소한의 품질 체크를 추가.
    """
    t = _clean_text(text)
    if not t:
        return False

    # 너무 짧으면 실패로 취급(속보도 있을 수 있어 낮게 잡음)
    if len(t) < 80:
        return False

    # 너무 반복적이거나(동일 문구 반복) 링크/버튼 텍스트가 많은 경우를 간단히 배제
    # (정교하게 하려면 더 많은 패턴/분포 분석이 필요)
    bad_markers = [
        "enable javascript",
        "cookies",
        "개인정보처리방침",
        "이용약관",
        "subscribe",
        "sign in",
        "로그인",
        "구독",
    ]
    lowered = t.lower()
    if any(m in lowered for m in bad_markers):
        # 단, 실제 기사에서도 "로그인"이 나올 수 있어 완전 차단은 위험하지만
        # 여기선 fetcher 품질 향상을 위해 보수적으로 처리.
        return False

    return True


# -----------------------------
# HTTP fetching (session + simple retry)
# -----------------------------
def _get(
    session: requests.Session,
    url: str,
    timeout_sec: int,
    max_redirects: int = 10,
) -> tuple[Optional[requests.Response], Optional[str]]:
    try:
        resp = session.get(
            url,
            headers=_make_headers(),
            timeout=timeout_sec,
            allow_redirects=True,
        )
        # requests는 내부에서 redirects 제한이 있지만, 별도 max_redirects 제어가 필요하면 adapter 커스텀 필요.
        return resp, None
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"


# -----------------------------
# Main API
# -----------------------------
def fetch_article_text(
    url: str,
    timeout_sec: int = 6,
    max_chars: int = 12000,
    session: Optional[requests.Session] = None,
) -> FetchResult:
    """
    리팩토링 포인트(왜 이렇게 작성했는지):
    - Google News: 상대경로 포함 후보 수집 → 내부 articles 페이지면 1회 더 열어 외부 원문 후보 탐색
    - HTTP status: 4xx/5xx는 본문 파싱 시도 전에 명확히 실패 처리(봇 차단/에러페이지 오염 방지)
    - 인코딩: apparent_encoding 기반 보정
    - 휴리스틱: "가장 긴 div" 대신 링크 비율/문장부호/길이 기반 점수로 후보 선택
    - 반환: meta를 dict/dataclass로 구조화(운영 로그/모니터링 용이)

    [주의]
    - 403/429는 사이트 정책/봇차단 이슈일 수 있음. 이 코드는 우회(프록시/쿠키/헤드리스)는 하지 않음.
    """
    owns_session = session is None
    s = session or requests.Session()
    notes: list[str] = []

    requested_url = url
    resp, err = _get(s, url, timeout_sec)
    if err or resp is None:
        meta = FetchMeta(
            requested_url=requested_url,
            final_url=url,
            status=0,
            html_len=0,
            extractor="none",
            notes=[f"request_error:{err or 'unknown'}"],
        )
        if owns_session:
            s.close()
        return FetchResult(text="", meta=meta)

    final_url = resp.url or url
    status = resp.status_code

    # 4xx/5xx는 본문 추출 시도 자체가 오염될 가능성이 높아서 즉시 실패 처리
    if status >= 400:
        block_hint = ""
        www_auth = (resp.headers.get("WWW-Authenticate") or "").lower()
        server_header = (resp.headers.get("Server") or "").lower()
        if "cloudflare" in server_header or "cf-ray" in (resp.headers.get("CF-RAY") or "").lower():
            block_hint = "blocked:cloudflare"
        elif "akamai" in server_header or "akamai" in (resp.headers.get("Akamai") or "").lower():
            block_hint = "blocked:akamai"
        elif "incapsula" in server_header or "imperva" in server_header:
            block_hint = "blocked:imperva"
        elif "bot" in www_auth or "captcha" in www_auth:
            block_hint = "blocked:auth_challenge"
        elif status in {401, 403, 429}:
            block_hint = f"blocked:http_{status}"

        meta = FetchMeta(
            requested_url=requested_url,
            final_url=final_url,
            status=status,
            html_len=len(resp.content or b""),
            extractor="none",
            notes=[f"http_error:{status}"] + ([block_hint] if block_hint else []),
        )
        if owns_session:
            s.close()
        return FetchResult(text="", meta=meta)

    ok, reason, raw = _response_body_ok(resp)
    if not ok:
        meta = FetchMeta(
            requested_url=requested_url,
            final_url=final_url,
            status=status,
            html_len=len(raw),
            extractor="none",
            notes=[reason],
        )
        if owns_session:
            s.close()
        return FetchResult(text="", meta=meta)

    html = _safe_decode_response(resp)
    html_len = len(html)

    # Google News면: canonical/og/refresh 우선, 없으면 a[href] 후보로 진행
    if _is_google_news(final_url):
        notes.append("google_news_detected")

        canonical = _extract_canonical_url(html, final_url)
        if canonical and canonical != final_url:
            notes.append("google_news_canonical_found")
            resp2, err2 = _get(s, canonical, timeout_sec)
            if resp2 is not None and not err2 and resp2.status_code < 400:
                ok2, reason2, raw2 = _response_body_ok(resp2)
                if ok2:
                    final_url = resp2.url or canonical
                    status = resp2.status_code
                    html = _safe_decode_response(resp2)
                    html_len = len(html)
                else:
                    notes.append(f"canonical_non_text:{reason2}")
            else:
                notes.append(f"canonical_fetch_failed:{err2 or (resp2.status_code if resp2 else 'no_resp')}")

        # 여전히 google news거나 본문이 없을 가능성이 커서,
        # 내부/외부 후보를 수집해서 외부 원문을 시도
        candidates = _extract_external_hrefs_from_google_news(html, final_url)
        best = _pick_best_external_url(candidates, final_url)

        if best and best != final_url:
            resp3, err3 = _get(s, best, timeout_sec)
            if resp3 is not None and not err3 and resp3.status_code < 400:
                ok3, reason3, raw3 = _response_body_ok(resp3)
                if not ok3:
                    notes.append(f"google_news_non_text:{reason3}")
                    ok3 = False
                if ok3:
                    html3 = _safe_decode_response(resp3)
                    final3 = resp3.url or best
                    notes.append("google_news_handoff_attempt")

                    # best가 여전히 news.google.com 내부면 한 번 더 외부를 찾는다
                    if _is_google_news(final3):
                        notes.append("google_news_internal_article_page")
                        candidates2 = _extract_external_hrefs_from_google_news(html3, final3)
                        best2 = _pick_best_external_url(candidates2, final3)
                        if best2 and best2 != final3:
                            resp4, err4 = _get(s, best2, timeout_sec)
                            if resp4 is not None and not err4 and resp4.status_code < 400:
                                ok4, reason4, raw4 = _response_body_ok(resp4)
                                if ok4:
                                    final_url = resp4.url or best2
                                    status = resp4.status_code
                                    html = _safe_decode_response(resp4)
                                    html_len = len(html)
                                    notes.append("google_news_external_resolved")
                                else:
                                    notes.append(f"google_news_external_non_text:{reason4}")
                                    # fallback: 내부 페이지라도 사용
                                    final_url, status, html, html_len = final3, resp3.status_code, html3, len(html3)
                            else:
                                notes.append(f"google_news_external_fetch_failed:{err4 or (resp4.status_code if resp4 else 'no_resp')}")
                                # fallback: 내부 페이지라도 사용
                                final_url, status, html, html_len = final3, resp3.status_code, html3, len(html3)
                        else:
                            # 외부 못 찾음: 내부 페이지라도 사용
                            final_url, status, html, html_len = final3, resp3.status_code, html3, len(html3)
                    else:
                        # 외부 페이지 확보 성공
                        final_url, status, html, html_len = final3, resp3.status_code, html3, len(html3)
            else:
                notes.append(f"google_news_best_candidate_fetch_failed:{err3 or (resp3.status_code if resp3 else 'no_resp')}")

    # 1) trafilatura
    text = _extract_with_trafilatura(final_url, html)
    extractor = "trafilatura"
    if not _looks_like_article_text(text):
        # 2) newspaper
        text2 = _extract_with_newspaper(final_url, html, timeout_sec)
        if _looks_like_article_text(text2):
            text = text2
            extractor = "newspaper"
        else:
            # 3) heuristic
            text3 = _extract_main_text_heuristic(html)
            text = text3
            extractor = "heuristic" if _looks_like_article_text(text3) else "none"

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        notes.append(f"truncated_to:{max_chars}")

    meta = FetchMeta(
        requested_url=requested_url,
        final_url=final_url,
        status=status,
        html_len=html_len,
        extractor=extractor,
        notes=notes,
    )

    if owns_session:
        s.close()

    return FetchResult(text=text if extractor != "none" else "", meta=meta)


# -----------------------------
# Convenience wrapper (compatible-ish)
# -----------------------------
def fetch_article_text_compat(
    url: str,
    timeout_sec: int = 6,
    max_chars: int = 12000,
) -> tuple[str, str]:
    """
    기존 시그니처(tuple[str, str])가 필요한 경우를 위한 래퍼.
    meta를 문자열로 뭉치던 방식 대신, 최소한 key=value 형태로 남긴다.
    """
    r = fetch_article_text(url, timeout_sec=timeout_sec, max_chars=max_chars)
    m = r.meta
    meta_str = (
        f"final_url={m.final_url}||status={m.status}||html_len={m.html_len}"
        f"||extractor={m.extractor}||notes={','.join(m.notes)}"
    )
    return r.text, meta_str
