from __future__ import annotations

from typing import Any, Callable, NotRequired, TypedDict


class PipelineItem(TypedDict, total=False):
    title: str
    link: str
    summary: str
    summaryRaw: str
    fullText: str
    published: NotRequired[str]
    score: float
    topic: str
    source: str
    impactSignals: list[str]
    dedupeKey: str
    matchedTo: NotRequired[str]
    readTimeSec: int
    ageHours: NotRequired[float]
    status: NotRequired[str]
    dropReason: NotRequired[str]
    ai: NotRequired[dict[str, Any]]


Item = PipelineItem
LogFunc = Callable[[str], None]
ParseFunc = Callable[[str], Any]

