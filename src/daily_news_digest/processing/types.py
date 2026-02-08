from __future__ import annotations

from typing import Any, Callable, NotRequired, TypedDict


class PipelineItem(TypedDict, total=False):
    title: str
    link: str
    summary: str
    summaryRaw: str
    fullText: str
    published: NotRequired[str]
    publishedAtUtc: NotRequired[str]
    updatedAtUtc: NotRequired[str]
    score: float
    topic: str
    source: str
    sourceRaw: NotRequired[str]
    impactSignals: list[str]
    dedupeKey: str
    dedupeKeyRule: NotRequired[str]
    dedupeKeyAI: NotRequired[str]
    clusterKey: NotRequired[str]
    clusterKeyRule: NotRequired[str]
    matchedTo: NotRequired[str]
    readTimeSec: int
    ageHours: NotRequired[float]
    status: NotRequired[str]
    dropReason: NotRequired[str]
    mergeReason: NotRequired[str]
    ai: NotRequired[dict[str, Any]]


Item = PipelineItem
LogFunc = Callable[[str], None]
ParseFunc = Callable[[str], Any]
