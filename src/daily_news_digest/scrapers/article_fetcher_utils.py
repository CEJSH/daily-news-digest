from __future__ import annotations

import base64
import re
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from daily_news_digest.utils.common import clean_text_ws

_WS_RE = re.compile(r"\s+")

_MEDIA_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".js",
    ".css",
    ".woff",
    ".woff2",
    ".ttf",
    ".mp4",
    ".webm",
    ".mp3",
    ".pdf",
)

_MEDIA_HOST_HINTS = (
    "gstatic.com",
    "googleusercontent.com",
)

_NON_ARTICLE_HOST_HINTS = (
    "accounts.google.com",
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
    "googletagservices.com",
    "g.doubleclick.net",
    "stats.g.doubleclick.net",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "polyfill.io",
    "weather.com",
    "accuweather.com",
    "wunderground.com",
    "weather.gov",
    "meteoblue.com",
    "weatherbug.com",
    "weatherapi.com",
    "forecast.io",
    "openweathermap.org",
    "weathernews.com",
    "uberproxy-pen-redirect.corp.google.com",
    "w3.org",
)

_NON_ARTICLE_PATH_HINTS = (
    "/analytics.js",
    "/gtm.js",
    "/gtag/js",
    "/log",
    "/collect",
    "/g/collect",
    "/pagead",
    "/signin",
    "/servicelogin",
    "/serviceLogin",
    "/v3/signin/identifier",
    "/accounts",
    "/license",
    "/licenses",
    "/licence",
    "/privacy",
    "/terms",
    "/policy",
    "/policies",
    "/legal",
    "/copyright",
    "/about",
    "/help",
    "/support",
    "/faq",
    "/contact",
)

_GOOGLE_NEWS_ARTICLE_RE = re.compile(r"https?://news\.google\.com/(?:rss/)?articles/([^/?]+)")

_JS_MARKERS = [
    "(function",
    "function(",
    "var ",
    "let ",
    "const ",
    "return ",
    "undefined",
    "window.",
    "document.",
    "=>",
]

_CSS_MARKERS = [
    "@font-face",
    "font-family:",
    "src: url(",
    "unicode-range:",
    "font-style:",
    "font-weight:",
]

_LICENSE_POLICY_MARKERS = [
    "permission is hereby granted",
    "the software is provided \"as is\"",
    "without warranty of any kind",
    "copyright (c)",
    "all rights reserved",
    "mit license",
    "apache license",
    "gnu general public license",
    "terms of service",
    "terms and conditions",
    "privacy policy",
    "cookie policy",
    "legal notice",
    "this privacy policy",
    "이용약관",
    "개인정보처리방침",
    "개인정보 처리방침",
    "쿠키 정책",
    "법적 고지",
]

_STRONG_LICENSE_MARKERS = [
    "permission is hereby granted",
    "the software is provided \"as is\"",
    "gnu general public license",
    "apache license version",
    "mit license",
]

_BAD_MARKERS = [
    "enable javascript",
    "cookies",
    "개인정보처리방침",
    "이용약관",
    "subscribe",
    "sign in",
    "로그인",
    "구독",
]


def clean_text(text: str) -> str:
    return clean_text_ws(text)


def normalize_url(base: str, href: str) -> str:
    return urljoin(base, href)


def is_google_news(url: str) -> bool:
    try:
        return urlparse(url).netloc.endswith("news.google.com")
    except Exception:
        return False


def is_probably_media_url(url: str) -> bool:
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


def is_probably_non_article_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    if host.endswith("news.google.com"):
        if path in ("", "/", "/home", "/home/") or path.startswith("/home"):
            return True
    if any(hint in host for hint in _NON_ARTICLE_HOST_HINTS):
        return True
    if host.endswith("w3.org") and path.startswith("/2000/svg"):
        return True
    if host.endswith("google.com"):
        if path.startswith("/log") or "format=json" in query:
            return True
    if any(hint in path for hint in _NON_ARTICLE_PATH_HINTS):
        return True
    return False


def decode_google_news_rss_url(url: str) -> str:
    m = _GOOGLE_NEWS_ARTICLE_RE.search(url or "")
    if not m:
        return ""
    token = m.group(1)
    if not token:
        return ""
    padded = token + "=" * (-len(token) % 4)
    try:
        data = base64.urlsafe_b64decode(padded)
    except Exception:
        return ""
    urls = re.findall(rb"https?://[^\x00-\x20\"'<>]+", data)
    if not urls:
        return ""
    for u in urls:
        try:
            s = u.decode("utf-8", "ignore")
        except Exception:
            continue
        if not is_google_news(s):
            return s
    try:
        return urls[0].decode("utf-8", "ignore")
    except Exception:
        return ""


def pick_redirect_target(redirect_urls: Sequence[str], final_url: str) -> str:
    urls: list[str] = []
    for u in redirect_urls:
        if u:
            urls.append(u)
    if final_url:
        urls.append(final_url)
    for u in urls:
        if u and not is_google_news(u) and not is_probably_media_url(u) and not is_probably_non_article_url(u):
            return u
    return ""


def extract_canonical_url(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return normalize_url(base_url, link.get("href"))

    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        return normalize_url(base_url, og.get("content"))

    refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
    if refresh and refresh.get("content"):
        m = re.search(r"url=(.+)", refresh["content"], flags=re.IGNORECASE)
        if m:
            return normalize_url(base_url, m.group(1).strip())

    return ""


def extract_external_hrefs_from_google_news(html: str, base_url: str, limit: int = 20) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = normalize_url(base_url, href)
        if abs_url.startswith("javascript:") or abs_url.startswith("#"):
            continue
        if is_probably_media_url(abs_url) or is_probably_non_article_url(abs_url):
            continue
        out.append(abs_url)
        if len(out) >= limit:
            break

    if len(out) < limit:
        for u in re.findall(r"https?://[^\s\"'>]+", html):
            if is_probably_media_url(u) or is_probably_non_article_url(u):
                continue
            out.append(u)
            if len(out) >= limit:
                break

    return _dedupe_keep_order(out)


def is_textual_content_type(content_type: str) -> bool:
    ct = (content_type or "").lower()
    if not ct:
        return True
    if any(x in ct for x in ["text/javascript", "application/javascript", "text/css"]):
        return False
    return any(x in ct for x in ["text/", "application/json", "application/xml", "application/xhtml+xml"])


def response_body_ok(content_type: str, url: str) -> tuple[bool, str]:
    if content_type and not is_textual_content_type(content_type):
        return False, f"non_textual_content_type:{content_type}"
    if is_probably_media_url(url):
        return False, "media_url_detected"
    return True, ""


def text_quality_score(tag) -> float:
    text = clean_text(tag.get_text(" "))
    if not text:
        return -1.0

    length = len(text)

    link_text_len = 0
    for a in tag.find_all("a"):
        link_text_len += len(clean_text(a.get_text(" ")))
    link_ratio = (link_text_len / max(1, length))

    punct = sum(text.count(c) for c in [".", "?", "!", "。", "！", "？"])
    punct_density = punct / max(1, length)

    if length < 120:
        return -1.0

    import math

    score = math.log(length)
    score += min(0.8, punct_density * 50)
    score -= link_ratio * 3.5

    if link_ratio > 0.35:
        score -= 2.0

    return score


def extract_main_text_heuristic(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "form"]):
        tag.decompose()
    for tag in soup.find_all(["header", "nav", "footer", "aside"]):
        tag.decompose()

    candidates = []

    for tag_name in ["article", "main", "section", "div"]:
        for tag in soup.find_all(tag_name):
            score = text_quality_score(tag)
            if score > 0:
                candidates.append((score, tag))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_tag = candidates[0][1]
        return clean_text(best_tag.get_text(" "))

    body = soup.body or soup
    return clean_text(body.get_text(" "))


def looks_like_article_text(text: str, min_chars: int = 80) -> bool:
    t = clean_text(text)
    if not t:
        return False

    if len(t) < min_chars:
        return False

    js_hits = sum(1 for m in _JS_MARKERS if m in t)
    if js_hits >= 3:
        return False

    css_hits = sum(1 for m in _CSS_MARKERS if m in t)
    if css_hits >= 3:
        return False

    hits = sum(1 for m in _LICENSE_POLICY_MARKERS if m in t)
    if hits >= 2:
        return False
    if any(m in t for m in _STRONG_LICENSE_MARKERS):
        return False

    lowered = t.lower()
    if any(m in lowered for m in _BAD_MARKERS):
        return False

    return True


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    uniq: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq
