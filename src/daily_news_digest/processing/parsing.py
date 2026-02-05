from __future__ import annotations

import re
from typing import Any, Callable

from daily_news_digest.utils import clean_text, trim_title_noise


SUMMARY_FALLBACK = "내용을 확인하려면 클릭하세요."


class EntryParser:
    def __init__(
        self,
        *,
        clean_text_func: Callable[[str], str] = clean_text,
        trim_title_noise_func: Callable[[str, str | None], str] = trim_title_noise,
    ) -> None:
        self._clean_text = clean_text_func
        self._trim_title_noise = trim_title_noise_func

    def strip_source_from_text(self, text: str, source_name: str) -> str:
        # 요약/제목 끝에 붙는 출처 표기를 제거한다.
        # 예: "제목 | 매체", "제목 - 매체", "제목매체" 같은 접미 출처를 정리해 UI 텍스트를 깔끔하게 만든다.
        if not text or not source_name:
            return text
        src = re.escape(source_name.strip())
        cleaned = re.sub(
            rf"(?:\s*[\|\-–—·•:｜ㅣ]\s*)?{src}\s*\.{{0,3}}\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(rf"\s+{src}\s*\.{{0,3}}\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"{src}\s*\.{{0,3}}\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def pick_analysis_text(self, full_text: str, summary_clean: str) -> str:
        # 분석용 텍스트는 본문이 있으면 본문을 쓰고, 없으면 정제된 요약을 사용한다.
        if full_text:
            return full_text
        return summary_clean or ""

    def _extract_full_text_parts(self, entry: Any) -> list[str]:
        # feed entry.content의 value들을 원본 순서대로 모아서 리스트로 반환한다.
        # 정제는 여기서 하지 않고, extract_full_text에서 일괄 처리한다.
        content_list = getattr(entry, "content", None)
        if isinstance(content_list, list) and content_list:
            parts: list[str] = []
            for content in content_list:
                value = ""
                if isinstance(content, dict):
                    value = content.get("value", "") or ""
                else:
                    value = getattr(content, "value", "") or ""
                if value:
                    parts.append(value)
            return parts
        return []

    def extract_full_text(self, entry: Any) -> str:
        # content 조각을 공백으로 합친 뒤 텍스트 정제를 수행해 full_text를 만든다.
        parts = self._extract_full_text_parts(entry)
        if parts:
            return self._clean_text(" ".join(parts))
        return ""

    def parse_entry(self, entry: Any, source_name: str) -> tuple[str, str, str, str, str, str, str]:
        # entry의 title/summary/content를 정제해 UI용 요약과 분석용 텍스트를 구성한다.
        # 반환값은 raw/clean/요약/본문/분석 텍스트를 일관된 순서로 제공한다.
        title_raw = getattr(entry, "title", "").strip()
        summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
        summary_raw = summary_raw or ""
        summary_clean = self._clean_text(summary_raw)
        summary_clean = self.strip_source_from_text(summary_clean, source_name)
        title_clean = self._trim_title_noise(self._clean_text(title_raw), source_name)
        summary = summary_clean if summary_clean else SUMMARY_FALLBACK
        parts = self._extract_full_text_parts(entry)
        full_text = self._clean_text(" ".join(parts)) if parts else ""
        analysis_text = self.pick_analysis_text(full_text, summary_clean)
        return (
            title_raw,
            title_clean,
            summary_raw,
            summary_clean,
            summary,
            full_text,
            analysis_text,
        )
