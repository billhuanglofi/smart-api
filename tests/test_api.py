"""Tests for the FastAPI application endpoints."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


def test_root_returns_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_list_models_no_auth():
    """Models endpoint should not require authentication."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    first = data["data"][0]
    assert "id" in first
    assert "object" in first
    assert first["object"] == "model"


def test_list_models_returns_known_model():
    resp = client.get("/v1/models")
    ids = [m["id"] for m in resp.json()["data"]]
    assert "gpt-4o" in ids


# ---------------------------------------------------------------------------
# /v1/chat/completions – auth errors
# ---------------------------------------------------------------------------


def test_chat_completions_missing_token_raises_401():
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with patch.dict("os.environ", {}, clear=False):
        import os

        original = os.environ.pop("GITHUB_TOKEN", None)
        try:
            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 401
        finally:
            if original is not None:
                os.environ["GITHUB_TOKEN"] = original


# ---------------------------------------------------------------------------
# /v1/chat/completions – happy path (mocked upstream)
# ---------------------------------------------------------------------------


MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def test_chat_completions_returns_openai_format():
    with patch(
        "src.main.chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.return_value = MOCK_CHAT_RESPONSE
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello there!"


def test_chat_completions_filters_profanity():
    dirty_response = dict(MOCK_CHAT_RESPONSE)
    dirty_response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a shit response.",
            },
            "finish_reason": "stop",
        }
    ]
    with patch(
        "src.main.chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.return_value = dirty_response
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "shit" not in content.lower()


# ---------------------------------------------------------------------------
# /v1/completions – happy path (mocked upstream)
# ---------------------------------------------------------------------------


def test_completions_returns_text_completion_format():
    with patch(
        "src.main.chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.return_value = MOCK_CHAT_RESPONSE
        payload = {
            "model": "gpt-4o",
            "prompt": "Tell me a joke",
        }
        resp = client.post(
            "/v1/completions",
            json=payload,
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]


def test_completions_filters_profanity():
    dirty_response = dict(MOCK_CHAT_RESPONSE)
    dirty_response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "What the hell is a joke?",
            },
            "finish_reason": "stop",
        }
    ]
    with patch(
        "src.main.chat_completions", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.return_value = dirty_response
        payload = {
            "model": "gpt-4o",
            "prompt": "Tell me a joke",
        }
        resp = client.post(
            "/v1/completions",
            json=payload,
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 200
    text = resp.json()["choices"][0]["text"]
    assert "hell" not in text.lower()
