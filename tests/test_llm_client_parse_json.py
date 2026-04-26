from __future__ import annotations

from daily_news_digest.processing.llm_client import parse_json


def test_parse_json_returns_none_for_empty() -> None:
    assert parse_json("") is None


def test_parse_json_handles_clean_object() -> None:
    assert parse_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_handles_code_fence() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_handles_trailing_commas_in_block() -> None:
    raw = 'preamble {"a": 1, "b": [1, 2,], } trailing'
    assert parse_json(raw) == {"a": 1, "b": [1, 2]}


def test_parse_json_preserves_json_keywords() -> None:
    assert parse_json('{"a": true, "b": false, "c": null}') == {
        "a": True,
        "b": False,
        "c": None,
    }


def test_parse_json_rejects_python_tuple_literal() -> None:
    """ast.literal_eval would have accepted this and returned a dict
    containing a tuple value, which then breaks JSON-aware downstream code."""
    assert parse_json('{"a": (1, 2)}') is None


def test_parse_json_rejects_python_only_keywords() -> None:
    """LLM is contracted to return JSON; Python keywords True/False/None
    indicate a malformed response and must not be silently accepted."""
    assert parse_json('{"a": True, "b": None}') is None


def test_parse_json_rejects_garbage() -> None:
    assert parse_json("this is not json at all") is None
