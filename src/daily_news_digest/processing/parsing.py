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
        return cleaned.strip()

    def pick_analysis_text(self, full_text: str, summary_clean: str) -> str:
        if full_text:
            return full_text
        return summary_clean or ""

    def extract_full_text(self, entry: Any) -> str:
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
            if parts:
                return self._clean_text(" ".join(parts))
        return ""

    def parse_entry(self, entry: Any, source_name: str) -> tuple[str, str, str, str, str, str, str]:
        title_raw = getattr(entry, "title", "").strip()
        summary_raw = getattr(entry, "summary", "") if hasattr(entry, "summary") else ""
        summary_clean = self._clean_text(summary_raw)
        summary_clean = self.strip_source_from_text(summary_clean, source_name)
        title_clean = self._trim_title_noise(self._clean_text(title_raw), source_name)
        summary = summary_clean if summary_clean else SUMMARY_FALLBACK
        full_text = self.extract_full_text(entry)
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

