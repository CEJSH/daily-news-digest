from __future__ import annotations

import atexit
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Sequence

from playwright.sync_api import BrowserContext
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from daily_news_digest.scrapers.article_fetcher_config import ArticleFetcherConfig
from daily_news_digest.scrapers.article_fetcher_utils import (
    clean_text,
    decode_google_news_rss_url,
    extract_canonical_url,
    extract_external_hrefs_from_google_news,
    extract_main_text_heuristic,
    is_google_news,
    is_probably_media_url,
    is_probably_non_article_url,
    looks_like_article_text,
    pick_redirect_target,
    response_body_ok,
)

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

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class PlaywrightResponse:
    url: str
    status: int
    headers: dict[str, str]
    html: str
    redirect_urls: list[str]


@dataclass(frozen=True)
class FetchError:
    kind: str
    message: str
    url: str

    def to_note(self) -> str:
        safe = (self.message or "").replace("\n", " ").replace("|", " ").strip()
        return f"request_error:{self.kind}:{safe}" if safe else f"request_error:{self.kind}"


# -----------------------------
# Playwright client
# -----------------------------
class PlaywrightClient:
    _pw = None
    _browser = None
    _lock = threading.Lock()

    def __init__(self, config: ArticleFetcherConfig, log: logging.Logger) -> None:
        self._config = config
        self._log = log
        self._contexts: list[BrowserContext] = []
        self._context_lock = threading.Lock()
        self._context_index = 0

    def _ensure_browser(self):
        with self._lock:
            if self._browser is None:
                self._pw = sync_playwright().start()
                self._browser = self._pw.chromium.launch(headless=self._config.headless)
        return self._browser

    def close(self) -> None:
        with self._lock:
            self._close_contexts()
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass
                self._pw = None

    def fetch(
        self,
        url: str,
        timeout_sec: int,
        referer: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
        wait_ms: int = 0,
    ) -> tuple[Optional[PlaywrightResponse], Optional[FetchError]]:
        context = None
        pooled = False
        page = None
        try:
            context, pooled = self._acquire_context()
            page = context.new_page()
            timeout_ms = max(1000, int(timeout_sec * 1000))
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)

            response = page.goto(url, wait_until="domcontentloaded", referer=referer)
            if response is None:
                html = page.content()
                final_url = page.url or url
                return (
                    PlaywrightResponse(
                        url=final_url,
                        status=0,
                        headers={},
                        html=html,
                        redirect_urls=[],
                    ),
                    None,
                )

            if wait_for_selector:
                try:
                    page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
                except Exception:
                    pass

            if wait_ms > 0:
                try:
                    page.wait_for_timeout(wait_ms)
                except Exception:
                    pass

            if self._config.post_nav_wait_ms > 0:
                try:
                    page.wait_for_timeout(self._config.post_nav_wait_ms)
                except Exception:
                    pass

            html = page.content()
            redirect_urls = self._collect_redirects(response)
            resp = PlaywrightResponse(
                url=response.url or page.url or url,
                status=response.status,
                headers={k.lower(): v for k, v in response.headers.items()},
                html=html,
                redirect_urls=redirect_urls,
            )
            return resp, None
        except PlaywrightTimeoutError as e:
            self._log.warning("Playwright timeout: %s", url)
            return None, FetchError("timeout", str(e), url)
        except PlaywrightError as e:
            self._log.warning("Playwright error: %s", url)
            return None, FetchError("playwright_error", str(e), url)
        except Exception as e:
            self._log.exception("Unexpected fetch error: %s", url)
            return None, FetchError(type(e).__name__, str(e), url)
        finally:
            try:
                if page is not None:
                    page.close()
            except Exception:
                pass
            if context is not None and not pooled:
                try:
                    context.close()
                except Exception:
                    pass

    def _pick_user_agent(self) -> str:
        return random.choice(self._config.user_agents)

    def _make_headers(self) -> dict[str, str]:
        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }
        return headers

    def _route_filter(self, route, request) -> None:
        if request.resource_type in set(self._config.block_resource_types):
            route.abort()
            return
        route.continue_()

    def _acquire_context(self) -> tuple[BrowserContext, bool]:
        if self._config.context_pool_size <= 0:
            return self._create_context(), False
        with self._context_lock:
            if len(self._contexts) < self._config.context_pool_size:
                ctx = self._create_context()
                self._contexts.append(ctx)
                return ctx, True
            ctx = self._contexts[self._context_index % len(self._contexts)]
            self._context_index += 1
            return ctx, True

    def _create_context(self) -> BrowserContext:
        browser = self._ensure_browser()
        headers = self._make_headers()
        context = browser.new_context(
            user_agent=self._pick_user_agent(),
            locale=self._config.browser_locale,
            extra_http_headers=headers,
            java_script_enabled=True,
        )
        context.route("**/*", self._route_filter)
        return context

    def _close_contexts(self) -> None:
        with self._context_lock:
            for ctx in self._contexts:
                try:
                    ctx.close()
                except Exception:
                    pass
            self._contexts = []

    @staticmethod
    def _collect_redirects(response) -> list[str]:
        redirect_urls: list[str] = []
        try:
            req = response.request
            while req is not None:
                if req.url:
                    redirect_urls.append(req.url)
                req = req.redirected_from
        except Exception:
            redirect_urls = []
        return redirect_urls


# -----------------------------
# Text extraction
# -----------------------------
class TextExtractor:
    def __init__(self, min_text_chars: int, user_agent: str = "Mozilla/5.0") -> None:
        self._min_text_chars = min_text_chars
        self._user_agent = user_agent

    def extract(self, url: str, html: str, timeout_sec: int) -> tuple[str, str]:
        text = self._extract_with_trafilatura(url, html)
        extractor = "trafilatura"
        if not looks_like_article_text(text, self._min_text_chars):
            text2 = self._extract_with_newspaper(url, html, timeout_sec)
            if looks_like_article_text(text2, self._min_text_chars):
                return text2, "newspaper"
            text3 = extract_main_text_heuristic(html)
            extractor = "heuristic" if looks_like_article_text(text3, self._min_text_chars) else "none"
            return text3, extractor
        return text, extractor

    @staticmethod
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
            return clean_text(extracted or "")
        except Exception:
            return ""

    def _extract_with_newspaper(self, url: str, html: str, timeout_sec: int) -> str:
        if Article is None:
            return ""
        try:
            config = None
            if NewspaperConfig is not None:
                config = NewspaperConfig()
                config.browser_user_agent = self._user_agent
                config.request_timeout = timeout_sec
                config.fetch_images = False
            article = Article(url, config=config) if config is not None else Article(url)
            try:
                article.download(input_html=html)
            except TypeError:
                article.download()
            article.parse()
            return clean_text(getattr(article, "text", "") or "")
        except Exception:
            return ""


# -----------------------------
# Main fetcher
# -----------------------------
class ArticleFetcher:
    def __init__(
        self,
        config: Optional[ArticleFetcherConfig] = None,
        log: Optional[logging.Logger] = None,
        client: Optional[PlaywrightClient] = None,
        extractor: Optional[TextExtractor] = None,
    ) -> None:
        self._config = config or ArticleFetcherConfig()
        self._log = log or logger
        self._client = client or PlaywrightClient(self._config, self._log)
        default_ua = random.choice(self._config.user_agents) if self._config.user_agents else "Mozilla/5.0"
        self._extractor = extractor or TextExtractor(self._config.min_text_chars, user_agent=default_ua)
        atexit.register(self._client.close)

    def fetch(self, url: str, timeout_sec: Optional[int] = None, max_chars: Optional[int] = None) -> FetchResult:
        notes: list[str] = []
        requested_url = url
        timeout = timeout_sec if timeout_sec is not None else self._config.default_timeout_sec
        max_len = max_chars if max_chars is not None else self._config.max_chars

        self._log.info("fetch_start: %s", url)

        resp, err = self._fetch_initial(url, timeout, notes)
        if err or resp is None:
            return self._build_error_result(requested_url, url, err, notes)

        final_url = resp.url or url
        status = resp.status

        if is_probably_non_article_url(final_url):
            notes.append(f"non_article_redirect:{final_url}")
            notes.append("login_or_policy_page")
            return self._build_meta_result(
                requested_url,
                final_url,
                status,
                len(resp.html or ""),
                "none",
                notes,
                text="",
            )

        if status >= 400:
            block_hint = self._detect_block_hint(resp.headers, status)
            notes.append(f"http_error:{status}")
            if block_hint:
                notes.append(block_hint)
            return self._build_meta_result(
                requested_url,
                final_url,
                status,
                len(resp.html or ""),
                "none",
                notes,
                text="",
            )

        ok, reason = response_body_ok(resp.headers.get("content-type", ""), final_url)
        if not ok:
            resp = self._retry_google_news_redirect(resp, timeout, notes) or resp
            ok, reason = response_body_ok(resp.headers.get("content-type", ""), resp.url)
            if not ok:
                notes.append(reason)
                return self._build_meta_result(
                    requested_url,
                    resp.url,
                    resp.status,
                    len(resp.html or ""),
                    "none",
                    notes,
                    text="",
                )

        html = resp.html or ""
        html_len = len(html)

        if is_google_news(resp.url):
            result = self._handle_google_news(resp.url, resp.status, html, timeout, notes)
            if result is None:
                return self._build_meta_result(
                    requested_url,
                    resp.url,
                    resp.status,
                    html_len,
                    "none",
                    notes + ["google_news_no_handoff"],
                    text="",
                )
            final_url, status, html, html_len = result

        text, extractor = self._extractor.extract(final_url, html, timeout)
        if max_len and len(text) > max_len:
            text = text[:max_len]
            notes.append(f"truncated_to:{max_len}")

        if extractor == "none":
            text = ""

        self._log.info("fetch_done: %s extractor=%s len=%s", final_url, extractor, len(text))

        return self._build_meta_result(
            requested_url,
            final_url,
            status,
            html_len,
            extractor,
            notes,
            text=text,
        )

    def fetch_many(
        self,
        urls: Sequence[str],
        timeout_sec: Optional[int] = None,
        max_chars: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> list[FetchResult]:
        if not urls:
            return []
        worker_count = max_workers if max_workers is not None else self._config.parallel_fetch_max_workers
        if worker_count <= 1:
            return [self.fetch(url, timeout_sec=timeout_sec, max_chars=max_chars) for url in urls]

        def _worker(target_url: str) -> FetchResult:
            worker = ArticleFetcher(config=self._config, log=self._log)
            return worker.fetch(target_url, timeout_sec=timeout_sec, max_chars=max_chars)

        results: list[FetchResult] = [None] * len(urls)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_worker, url): idx for idx, url in enumerate(urls)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    self._log.exception("fetch_many error: %s", exc)
                    results[idx] = FetchResult(
                        text="",
                        meta=FetchMeta(
                            requested_url=urls[idx],
                            final_url=urls[idx],
                            status=0,
                            html_len=0,
                            extractor="none",
                            notes=[f"request_error:parallel:{type(exc).__name__}"],
                        ),
                    )
        return results

    def _fetch_initial(
        self,
        url: str,
        timeout: int,
        notes: list[str],
    ) -> tuple[Optional[PlaywrightResponse], Optional[FetchError]]:
        google_referer = self._config.google_news_referer if is_google_news(url) else None

        if is_google_news(url):
            decoded_url = decode_google_news_rss_url(url)
            if decoded_url and not is_probably_media_url(decoded_url) and not is_probably_non_article_url(decoded_url):
                notes.append("google_news_rss_decoded")
                self._log.info("google_news_rss_decoded: %s", decoded_url)
                resp, err = self._fetch_with_google_news_wait(
                    decoded_url,
                    timeout,
                    referer=google_referer,
                )
                if resp is not None and not err:
                    return resp, err
                notes.append(f"google_news_rss_decode_fetch_failed:{err.to_note() if err else 'no_resp'}")
            elif decoded_url:
                notes.append(f"google_news_rss_decoded_skipped:non_article:{decoded_url}")

        if is_google_news(url):
            return self._fetch_with_google_news_wait(url, timeout, referer=google_referer)
        return self._client.fetch(url, timeout, referer=google_referer)

    def _retry_google_news_redirect(
        self,
        resp: PlaywrightResponse,
        timeout: int,
        notes: list[str],
    ) -> Optional[PlaywrightResponse]:
        if not is_google_news(resp.url):
            return None
        alt = pick_redirect_target(resp.redirect_urls, resp.url)
        if alt:
            notes.append(f"google_news_redirect_candidate:{alt}")
            self._log.info("google_news_redirect_candidate: %s", alt)
            if is_probably_media_url(alt) or is_probably_non_article_url(alt):
                notes.append(f"google_news_redirect_skipped:non_article:{alt}")
                return None
            resp_alt, err_alt = self._fetch_with_google_news_wait(
                alt,
                timeout,
                referer=resp.url,
            )
            if resp_alt is not None and not err_alt and resp_alt.status < 400:
                ok_alt, reason_alt = response_body_ok(
                    resp_alt.headers.get("content-type", ""),
                    resp_alt.url,
                )
                if ok_alt:
                    return resp_alt
                notes.append(f"google_news_redirect_non_text:{reason_alt}")
        return None

    def _handle_google_news(
        self,
        final_url: str,
        status: int,
        html: str,
        timeout: int,
        notes: list[str],
    ) -> Optional[tuple[str, int, str, int]]:
        notes.append("google_news_detected")

        canonical = extract_canonical_url(html, final_url)
        if canonical and canonical != final_url:
            notes.append("google_news_canonical_found")
            self._log.info("google_news_canonical: %s", canonical)
            if is_probably_media_url(canonical) or is_probably_non_article_url(canonical):
                notes.append(f"canonical_non_article_url:{canonical}")
            else:
                resp2, err2 = self._client.fetch(canonical, timeout, referer=final_url)
                if resp2 is not None and not err2 and resp2.status < 400:
                    ok2, reason2 = response_body_ok(
                        resp2.headers.get("content-type", ""),
                        resp2.url,
                    )
                    if ok2:
                        final_url = resp2.url or canonical
                        status = resp2.status
                        html = resp2.html or ""
                    else:
                        notes.append(f"canonical_non_text:{reason2}")
                else:
                    notes.append(f"canonical_fetch_failed:{err2.to_note() if err2 else 'no_resp'}")

        candidates = extract_external_hrefs_from_google_news(
            html,
            final_url,
            limit=self._config.candidate_limit,
        )
        notes.append(f"google_news_candidates:{len(candidates)}")
        if candidates:
            sample = candidates[: min(5, len(candidates))]
            notes.append(f"google_news_candidates_sample:{'|'.join(sample)}")
        self._log_candidates("google_news_hrefs", candidates)

        result = self._attempt_candidates(candidates, final_url, timeout, notes, depth=0)
        if result:
            return result

        if is_google_news(final_url):
            return None
        return final_url, status, html, len(html)

    def _attempt_candidates(
        self,
        candidates: list[str],
        current: str,
        timeout: int,
        notes: list[str],
        depth: int,
    ) -> Optional[tuple[str, int, str, int]]:
        if not candidates:
            return None
        non_google = [u for u in candidates if not is_google_news(u)]
        google = [u for u in candidates if is_google_news(u)]
        ordered = non_google + google
        notes.append(
            f"attempt_candidates(depth={depth}): total={len(ordered)}, "
            f"non_google={len(non_google)}, google={len(google)}"
        )
        self._log_candidates(f"google_news_candidates_depth_{depth}", ordered)

        for cand in ordered:
            notes.append(f"attempt_candidates_try:{cand}")
            self._log.info("google_news_candidate_try: %s", cand)
            if cand == current:
                continue
            if is_probably_media_url(cand) or is_probably_non_article_url(cand):
                notes.append(f"google_news_candidate_skipped:non_article:{cand}")
                continue
            if is_google_news(cand):
                resp_c, err_c = self._fetch_with_google_news_wait(
                    cand,
                    timeout,
                    referer=current,
                )
            else:
                resp_c, err_c = self._client.fetch(cand, timeout, referer=current)
            if resp_c is None or err_c or resp_c.status >= 400:
                notes.append(
                    f"google_news_candidate_fetch_failed:{err_c.to_note() if err_c else resp_c.status if resp_c else 'no_resp'}"
                )
                continue
            ok_c, reason_c = response_body_ok(
                resp_c.headers.get("content-type", ""),
                resp_c.url,
            )
            if not ok_c:
                notes.append(f"google_news_non_text:{reason_c}")
                if "javascript" in reason_c:
                    continue
                continue
            html_c = resp_c.html or ""
            final_c = resp_c.url or cand
            notes.append("google_news_handoff_attempt")
            if is_google_news(final_c):
                if depth < self._config.max_candidate_depth:
                    notes.append("google_news_internal_article_page")
                    candidates2 = extract_external_hrefs_from_google_news(
                        html_c,
                        final_c,
                        limit=self._config.candidate_limit,
                    )
                    notes.append(f"google_news_candidates_depth_{depth+1}:{len(candidates2)}")
                    if candidates2:
                        sample2 = candidates2[: min(5, len(candidates2))]
                        notes.append(f"google_news_candidates_depth_{depth+1}_sample:{'|'.join(sample2)}")
                    self._log_candidates("google_news_hrefs_depth_1", candidates2)
                    result = self._attempt_candidates(candidates2, final_c, timeout, notes, depth + 1)
                    if result:
                        return result
                continue
            return final_c, resp_c.status, html_c, len(html_c)
        return None

    def _fetch_with_google_news_wait(
        self,
        url: str,
        timeout: int,
        referer: Optional[str],
    ) -> tuple[Optional[PlaywrightResponse], Optional[FetchError]]:
        return self._client.fetch(
            url,
            timeout,
            referer=referer,
            wait_for_selector=self._config.google_news_wait_for_selector,
            wait_ms=self._config.google_news_wait_ms,
        )

    def _detect_block_hint(self, headers: dict[str, str], status: int) -> str:
        server_header = (headers.get("server") or "").lower()
        www_auth = (headers.get("www-authenticate") or "").lower()
        if "cloudflare" in server_header or "cf-ray" in (headers.get("cf-ray") or "").lower():
            return "blocked:cloudflare"
        if "akamai" in server_header:
            return "blocked:akamai"
        if "incapsula" in server_header or "imperva" in server_header:
            return "blocked:imperva"
        if "bot" in www_auth or "captcha" in www_auth:
            return "blocked:auth_challenge"
        if status in {401, 403, 429}:
            return f"blocked:http_{status}"
        return ""

    def _log_candidates(self, label: str, candidates: Sequence[str]) -> None:
        if not self._config.log_candidates:
            return
        if not candidates:
            self._log.info("%s candidates: none", label)
            return
        limit = self._config.candidate_log_limit
        if len(candidates) <= limit:
            self._log.info("%s candidates(%s): %s", label, len(candidates), " | ".join(candidates))
            return
        head = candidates[:limit]
        self._log.info(
            "%s candidates(%s) first_%s: %s",
            label,
            len(candidates),
            limit,
            " | ".join(head),
        )

    def _build_error_result(
        self,
        requested_url: str,
        final_url: str,
        err: Optional[FetchError],
        notes: list[str],
    ) -> FetchResult:
        meta_notes = list(notes)
        if err:
            meta_notes.append(err.to_note())
        else:
            meta_notes.append("request_error:unknown")
        return self._build_meta_result(
            requested_url,
            final_url,
            0,
            0,
            "none",
            meta_notes,
            text="",
        )

    @staticmethod
    def _build_meta_result(
        requested_url: str,
        final_url: str,
        status: int,
        html_len: int,
        extractor: str,
        notes: list[str],
        text: str,
    ) -> FetchResult:
        meta = FetchMeta(
            requested_url=requested_url,
            final_url=final_url,
            status=status,
            html_len=html_len,
            extractor=extractor,
            notes=notes,
        )
        return FetchResult(text=text, meta=meta)


_DEFAULT_FETCHER = ArticleFetcher()


# -----------------------------
# Public API wrappers
# -----------------------------

def fetch_article_text(
    url: str,
    timeout_sec: Optional[int] = None,
    max_chars: Optional[int] = None,
    session: Optional[object] = None,
) -> FetchResult:
    _ = session
    return _DEFAULT_FETCHER.fetch(url, timeout_sec=timeout_sec, max_chars=max_chars)
