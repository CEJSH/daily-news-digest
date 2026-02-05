from __future__ import annotations

from typing import NotRequired, TypedDict


class ImpactSignal(TypedDict):
    label: str
    evidence: str


class DigestItem(TypedDict):
    id: str
    date: str
    category: str
    title: str
    summary: list[str]
    whyImportant: str
    importanceRationale: str
    impactSignals: list[ImpactSignal]
    sourceName: str
    sourceUrl: str
    status: str
    importance: int
    qualityLabel: str
    qualityReason: str
    matchedTo: NotRequired[str]
    publishedAt: NotRequired[str]
    readTimeSec: NotRequired[int]
    dedupeKey: NotRequired[str]
    clusterKey: NotRequired[str]
    dropReason: NotRequired[str]


class DailyDigest(TypedDict):
    date: str
    selectionCriteria: str
    editorNote: str
    question: str
    lastUpdatedAt: str
    items: list[DigestItem]
