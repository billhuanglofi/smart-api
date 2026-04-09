"""GitHub Copilot API client.

Handles authentication (GitHub token → short-lived Copilot session token)
and forwards chat/completion requests to the upstream Copilot service.
"""

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

# GitHub endpoints
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Copilot service endpoints
COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_CHAT_URL = f"{COPILOT_BASE_URL}/chat/completions"

# User-agent string that mimics the VSCode Copilot extension
_USER_AGENT = (
    "GitHubCopilotChat/0.22.4 vscode/1.89.0 vscode-insiders/1.89.0"
    " node/20.14.0 x64 linux"
)

# In-memory token cache – keyed by github_token
_token_cache: dict[str, dict[str, Any]] = {}
# Per-token lock to prevent duplicate token fetches under concurrent load
_token_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


async def _get_token_lock(github_token: str) -> asyncio.Lock:
    """Return (and lazily create) the asyncio.Lock for *github_token*."""
    async with _locks_lock:
        if github_token not in _token_locks:
            _token_locks[github_token] = asyncio.Lock()
        return _token_locks[github_token]


async def _fetch_copilot_token(github_token: str) -> str:
    """Exchange *github_token* for a short-lived Copilot API token."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(_COPILOT_TOKEN_URL, headers=headers)
        response.raise_for_status()
        data = response.json()

    token: str = data["token"]
    # The API returns an expiry timestamp; we refresh 60 s before it expires.
    expires_at: float = float(data.get("expires_at", time.time() + 1800)) - 60
    _token_cache[github_token] = {"token": token, "expires_at": expires_at}
    return token


async def get_copilot_token(github_token: str) -> str:
    """Return a valid Copilot session token, refreshing if necessary.

    A per-token :class:`asyncio.Lock` ensures that only one coroutine
    fetches a new token at a time even under concurrent requests.
    """
    # Fast path: check without holding the lock
    cached = _token_cache.get(github_token)
    if cached and time.time() < cached["expires_at"]:
        return cached["token"]

    lock = await _get_token_lock(github_token)
    async with lock:
        # Re-check inside the lock (another coroutine may have refreshed)
        cached = _token_cache.get(github_token)
        if cached and time.time() < cached["expires_at"]:
            return cached["token"]
        return await _fetch_copilot_token(github_token)


def _make_request_headers(copilot_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {copilot_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": "vscode/1.89.0",
        "Editor-Plugin-Version": "copilot-chat/0.22.4",
        "OpenAI-Intent": "conversation-panel",
        "X-Request-Id": str(uuid.uuid4()),
    }


async def chat_completions(
    github_token: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send a non-streaming chat completion request and return the full response."""
    copilot_token = await get_copilot_token(github_token)
    headers = _make_request_headers(copilot_token)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(COPILOT_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def stream_chat_completions(
    github_token: str,
    payload: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Yield raw SSE lines from a streaming chat completion request."""
    copilot_token = await get_copilot_token(github_token)
    headers = _make_request_headers(copilot_token)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", COPILOT_CHAT_URL, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line
