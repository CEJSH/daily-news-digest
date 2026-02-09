"""Impact Signal 처리 및 검증 함수들."""
from __future__ import annotations

import re
from typing import Any

from daily_news_digest.utils import clean_text
from daily_news_digest.export.constants import ALLOWED_IMPACT_LABELS
from daily_news_digest.export.validators.evidence import (
    label_evidence_valid,
    evidence_specificity_score,
)


def normalize_evidence_key(text: str) -> str:
    """Evidence 중복 검사용 키 생성."""
    t = clean_text(text or "").lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_evidence_too_short(text: str) -> bool:
    """Evidence 길이가 너무 짧은지 확인."""
    t = clean_text(text or "")
    if not t:
        return True
    if len(t) < 20:
        return True
    if len(t.split()) < 6:
        return True
    return False


def has_duplicate_impact_labels(impact_signals: Any) -> bool:
    """중복된 impact signal 라벨이 있는지 확인."""
    if not isinstance(impact_signals, list):
        return False
    labels: list[str] = []
    for entry in impact_signals:
        if isinstance(entry, dict):
            label = clean_text(entry.get("label") or "").lower()
        else:
            label = clean_text(str(entry)).lower()
        if not label:
            continue
        labels.append(label)
    return len(set(labels)) != len(labels)


def has_duplicate_impact_evidence(impact_signals: Any) -> bool:
    """중복된 impact signal evidence가 있는지 확인."""
    if not isinstance(impact_signals, list):
        return False
    seen: set[str] = set()
    for entry in impact_signals:
        if not isinstance(entry, dict):
            continue
        evidence = clean_text(entry.get("evidence") or "")
        if not evidence:
            continue
        key = normalize_evidence_key(evidence)
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def sanitize_impact_signals(raw: Any, full_text: str, summary_text: str) -> list[dict[str, str]]:
    """Impact signal 목록 정제 (유효성 검증, 중복 제거, 최적 evidence 선택)."""
    if not isinstance(raw, list):
        return []
    source_text = full_text or summary_text or ""
    source_norm = clean_text(source_text or "").lower()
    if not source_norm:
        return []
    candidates: dict[str, tuple[tuple[int, int, int], str]] = {}
    
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = clean_text(entry.get("label") or "").lower()
        evidence = clean_text(entry.get("evidence") or "")
        if label not in ALLOWED_IMPACT_LABELS:
            continue
        if not evidence:
            continue
        if is_evidence_too_short(evidence):
            continue
        if clean_text(evidence).lower() not in source_norm:
            continue
        if not label_evidence_valid(label, evidence):
            continue
        score = evidence_specificity_score(label, evidence)
        best = candidates.get(label)
        if not best or score > best[0]:
            candidates[label] = (score, evidence)
            
    ordered: list[tuple[str, tuple[int, int, int], str]] = [
        (label, payload[0], payload[1]) for label, payload in candidates.items()
    ]
    ordered.sort(key=lambda x: x[1], reverse=True)

    cleaned: list[dict[str, str]] = []
    seen_evidence: set[str] = set()
    for label, _score, evidence in ordered:
        evidence_key = normalize_evidence_key(evidence)
        if not evidence_key or evidence_key in seen_evidence:
            continue
        cleaned.append({"label": label, "evidence": evidence})
        seen_evidence.add(evidence_key)
    return cleaned
