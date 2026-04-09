"""Tests for the content filter module."""

import pytest
from src.filter import contains_profanity, filter_text


def test_filter_clean_text_unchanged():
    text = "Hello, how are you today?"
    assert filter_text(text) == text


def test_filter_removes_profanity():
    # Use a known profane word; the filter replaces each character with '*'
    dirty = "This is a shit example."
    result = filter_text(dirty)
    assert "shit" not in result.lower()
    # The rest of the sentence structure should be preserved
    assert "This is a" in result
    assert "example." in result


def test_filter_empty_string():
    assert filter_text("") == ""


def test_filter_none_returns_none():
    assert filter_text(None) is None


def test_contains_profanity_true():
    assert contains_profanity("What the hell is going on?") is True


def test_contains_profanity_false():
    assert contains_profanity("The weather is nice today.") is False


def test_filter_preserves_code():
    code = "def foo():\n    return 42"
    assert filter_text(code) == code


def test_filter_multiple_bad_words():
    text = "shit and damn together"
    result = filter_text(text)
    assert "shit" not in result.lower()
    assert "damn" not in result.lower()
