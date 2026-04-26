from __future__ import annotations

from typing import Any

from daily_news_digest.processing.pipeline import DigestPipeline


class _FakeFilterScorer:
    """pick_top_with_mix 가 호출하는 메소드들의 최소 구현."""

    def is_eligible(self, item: dict) -> bool:
        return not item.get("dropReason") and item.get("status") != "merged"

    def passes_top_freshness(self, *_args, **_kwargs) -> bool:
        return True

    def get_item_category(self, item: dict) -> str:
        return item.get("category") or "기타"

    def get_long_impact_labels(self, _text: str, _signals: list) -> set:
        return set()

    def is_top_source_allowed(self, _source: str | None, _source_raw: str | None) -> bool:
        return True


class _FakeAIService:
    def __init__(self) -> None:
        self.prefetch_calls: list[list[dict]] = []
        self.fail_links: set[str] = set()

    def prefetch_full_text(self, items: list[dict]) -> None:
        self.prefetch_calls.append(list(items))
        for it in items:
            if it.get("link") in self.fail_links:
                it["dropReason"] = "fulltext_missing"


def _build_pipeline(scorer: _FakeFilterScorer, ai: _FakeAIService) -> DigestPipeline:
    pipe = DigestPipeline.__new__(DigestPipeline)
    pipe._filter_scorer = scorer
    pipe._ai_service = ai
    pipe._log = lambda msg: None
    return pipe


def _item(score: float, link: str) -> dict[str, Any]:
    return {
        "score": score,
        "link": link,
        "title": link,
        "summary": "",
        "fullText": "",
        "source": "src",
        "category": "기타",
        "impactSignals": [],
    }


def test_refill_returns_eligible_picks_when_all_pass() -> None:
    scorer = _FakeFilterScorer()
    ai = _FakeAIService()
    pipe = _build_pipeline(scorer, ai)
    picks = [_item(5, "a"), _item(4, "b"), _item(3, "c")]
    all_items = list(picks) + [_item(2, "d"), _item(1, "e")]

    out = pipe._refill_after_prefetch(picks, all_items, top_limit=3)

    assert [it["link"] for it in out] == ["a", "b", "c"]
    assert ai.prefetch_calls == [], "이미 모두 자격을 갖췄으면 추가 prefetch 가 없어야 함"


def test_refill_keeps_eligible_picks_and_fills_only_missing_slots() -> None:
    scorer = _FakeFilterScorer()
    ai = _FakeAIService()
    ai.fail_links = {"b"}
    pipe = _build_pipeline(scorer, ai)
    picks = [_item(5, "a"), _item(4, "b"), _item(3, "c")]
    extras = [_item(2, "d"), _item(1, "e")]
    all_items = list(picks) + extras

    ai.prefetch_full_text(picks)

    out = pipe._refill_after_prefetch(picks, all_items, top_limit=3)
    out_links = [it["link"] for it in out]

    assert "a" in out_links and "c" in out_links, "본문 확보된 기존 픽은 유지되어야 함"
    assert "b" not in out_links, "본문 실패한 항목은 빠져야 함"
    assert len(out) == 3, "부족분만큼 채워야 함"
    refill_target = out_links[-1]
    assert refill_target in {"d", "e"}, "refill 은 새로운 후보 풀에서 와야 함"


def test_refill_does_not_duplicate_existing_kept_items() -> None:
    scorer = _FakeFilterScorer()
    ai = _FakeAIService()
    ai.fail_links = {"a"}
    pipe = _build_pipeline(scorer, ai)
    picks = [_item(5, "a"), _item(4, "b")]
    all_items = list(picks) + [_item(3, "c")]

    ai.prefetch_full_text(picks)

    out = pipe._refill_after_prefetch(picks, all_items, top_limit=2)
    out_links = [it["link"] for it in out]

    assert out_links.count("b") == 1, "유지된 픽이 refill 결과에서 중복되면 안 됨"
    assert "c" in out_links
