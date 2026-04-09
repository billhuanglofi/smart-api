"""Smart API – OpenAI-compatible local server wrapping GitHub Copilot.

Start the server::

    uvicorn src.main:app --host 0.0.0.0 --port 8000

Set the ``GITHUB_TOKEN`` environment variable (or place it in a ``.env``
file) before starting.  The token must belong to a GitHub account that has
an active GitHub Copilot subscription.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.copilot import chat_completions, stream_chat_completions
from src.filter import filter_text

load_dotenv()

app = FastAPI(
    title="Smart API",
    description="OpenAI-compatible local API server wrapping GitHub Copilot",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
    "copilot-chat",
]


def _resolve_github_token(authorization: Optional[str]) -> str:
    """Return the GitHub token from the Authorization header or env var.

    The header must be in the form ``Bearer <token>`` or ``token <token>``.
    Falls back to the ``GITHUB_TOKEN`` environment variable.
    """
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise HTTPException(
            status_code=401,
            detail=(
                "GitHub token is required.  Supply it via the Authorization "
                "header (Bearer <token>) or the GITHUB_TOKEN environment variable."
            ),
        )
    return token


def _filter_message_content(content: Any) -> Any:
    """Recursively filter profanity from message content.

    *content* can be a plain string, a list of content parts (OpenAI vision
    format), or ``None``.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return filter_text(content)
    if isinstance(content, list):
        filtered = []
        for part in content:
            if isinstance(part, dict):
                p = dict(part)
                if "text" in p:
                    p["text"] = filter_text(p["text"])
                filtered.append(p)
            else:
                filtered.append(part)
        return filtered
    return content


def _filter_response(data: dict[str, Any]) -> dict[str, Any]:
    """Filter profanity from every ``message.content`` field in *data*."""
    choices = data.get("choices", [])
    for choice in choices:
        message = choice.get("message")
        if message and "content" in message:
            message["content"] = _filter_message_content(message["content"])
        # Also filter delta content used in streaming chunks
        delta = choice.get("delta")
        if delta and "content" in delta:
            delta["content"] = _filter_message_content(delta["content"])
    return data


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: Any  # str | list[dict] for vision messages
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: list[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: bool = False
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "gpt-4o"
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: bool = False
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "message": "Smart API is running"}


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    """Return available models in OpenAI format."""
    now = int(time.time())
    models = [
        {
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "github-copilot",
        }
        for model_id in _SUPPORTED_MODELS
    ]
    return JSONResponse({"object": "list", "data": models})


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
) -> Any:
    """OpenAI-compatible chat completions endpoint backed by GitHub Copilot."""
    github_token = _resolve_github_token(authorization)

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                **({"name": m.name} if m.name else {}),
            }
            for m in request.messages
        ],
        "stream": request.stream,
    }
    # Forward optional parameters only when set by the caller
    for field in (
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "user",
    ):
        value = getattr(request, field)
        if value is not None:
            payload[field] = value

    if request.stream:
        return StreamingResponse(
            _stream_chat(github_token, payload),
            media_type="text/event-stream",
        )

    try:
        data = await chat_completions(github_token, payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc

    return JSONResponse(_filter_response(data))


async def _stream_chat(github_token: str, payload: dict[str, Any]):
    """Async generator that yields filtered SSE lines."""
    try:
        async for line in stream_chat_completions(github_token, payload):
            if not line:
                yield "\n"
                continue

            if line.startswith("data: "):
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    chunk = json.loads(raw)
                    chunk = _filter_response(chunk)
                    yield f"data: {json.dumps(chunk)}\n\n"
                except json.JSONDecodeError:
                    yield f"{line}\n\n"
            else:
                yield f"{line}\n\n"
    except httpx.HTTPStatusError as exc:
        error_payload = json.dumps(
            {
                "error": {
                    "message": exc.response.text,
                    "type": "upstream_error",
                    "code": exc.response.status_code,
                }
            }
        )
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None),
) -> Any:
    """OpenAI-compatible text completions endpoint (wraps chat completions)."""
    github_token = _resolve_github_token(authorization)

    # Translate the legacy completion request into a chat message
    chat_payload: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": request.stream,
    }
    for field in (
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "user",
    ):
        value = getattr(request, field)
        if value is not None:
            chat_payload[field] = value

    if request.stream:
        return StreamingResponse(
            _stream_completions(github_token, chat_payload),
            media_type="text/event-stream",
        )

    try:
        chat_data = await chat_completions(github_token, chat_payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=exc.response.text,
        ) from exc

    # Convert the chat response back to a legacy completion response
    completion_data = _chat_to_completion(chat_data, request.model)
    return JSONResponse(_filter_completion_response(completion_data))


def _chat_to_completion(chat_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a chat completion response to a legacy text completion format."""
    choices = []
    for choice in chat_data.get("choices", []):
        message = choice.get("message", {})
        choices.append(
            {
                "text": message.get("content", ""),
                "index": choice.get("index", 0),
                "logprobs": choice.get("logprobs"),
                "finish_reason": choice.get("finish_reason"),
            }
        )
    return {
        "id": chat_data.get("id", f"cmpl-{uuid.uuid4().hex}"),
        "object": "text_completion",
        "created": chat_data.get("created", int(time.time())),
        "model": model,
        "choices": choices,
        "usage": chat_data.get("usage"),
    }


def _filter_completion_response(data: dict[str, Any]) -> dict[str, Any]:
    """Filter profanity from legacy text completion choices."""
    for choice in data.get("choices", []):
        if "text" in choice:
            choice["text"] = filter_text(choice["text"])
    return data


async def _stream_completions(github_token: str, payload: dict[str, Any]):
    """Async generator that yields filtered SSE lines in completion format."""
    try:
        async for line in stream_chat_completions(github_token, payload):
            if not line:
                yield "\n"
                continue

            if line.startswith("data: "):
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    chunk = json.loads(raw)
                    # Remap delta content to text choices
                    new_choices = []
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        text = delta.get("content", "")
                        new_choices.append(
                            {
                                "text": filter_text(text) if text else "",
                                "index": choice.get("index", 0),
                                "logprobs": choice.get("logprobs"),
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )
                    completion_chunk = {
                        "id": chunk.get("id", f"cmpl-{uuid.uuid4().hex}"),
                        "object": "text_completion",
                        "created": chunk.get("created", int(time.time())),
                        "model": chunk.get("model", ""),
                        "choices": new_choices,
                    }
                    yield f"data: {json.dumps(completion_chunk)}\n\n"
                except json.JSONDecodeError:
                    yield f"{line}\n\n"
            else:
                yield f"{line}\n\n"
    except httpx.HTTPStatusError as exc:
        error_payload = json.dumps(
            {
                "error": {
                    "message": exc.response.text,
                    "type": "upstream_error",
                    "code": exc.response.status_code,
                }
            }
        )
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"
