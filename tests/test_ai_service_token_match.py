from __future__ import annotations

from daily_news_digest.processing.ai_service import AIEnrichmentService


def _service() -> AIEnrichmentService:
    return AIEnrichmentService.__new__(AIEnrichmentService)


def test_contains_incremental_token_matches_alpha_token_with_word_boundary() -> None:
    svc = _service()
    assert svc._contains_incremental_token("the policy was approved", "approved") is True


def test_contains_incremental_token_rejects_substring_of_larger_word() -> None:
    svc = _service()
    assert svc._contains_incremental_token("unapproved request", "approved") is False


def test_contains_incremental_token_matches_token_at_string_start() -> None:
    svc = _service()
    assert svc._contains_incremental_token("approved", "approved") is True


def test_contains_incremental_token_non_alpha_uses_substring() -> None:
    svc = _service()
    assert svc._contains_incremental_token("총 1조원 규모", "1조원") is True
