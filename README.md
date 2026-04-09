# smart-api

OpenAI-compatible local API server that wraps **GitHub Copilot** so you can use
any OpenAI-SDK-compatible tool with your existing Copilot subscription.
Responses are automatically filtered to remove profanity and offensive language.

---

## Features

- **OpenAI-compatible API** – drop-in replacement for OpenAI endpoints
  (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- **Streaming support** – server-sent events (SSE) for real-time token delivery
- **Content filter** – profanity and offensive words are replaced before the
  response reaches the client
- **GitHub Copilot backend** – uses your GitHub account's Copilot subscription;
  no separate API key required

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 recommended |
| GitHub account | with an active [GitHub Copilot](https://github.com/features/copilot) subscription |
| GitHub Personal Access Token | create at <https://github.com/settings/tokens> |

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/billhuanglofi/smart-api.git
cd smart-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your GitHub token
cp .env.example .env
# Edit .env and set GITHUB_TOKEN=ghp_…

# 4. Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The server listens on `http://localhost:8000` by default.

---

## Authentication

Pass your GitHub token as a Bearer token **or** set `GITHUB_TOKEN` in `.env`:

```bash
# via Authorization header
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ghp_your_token" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'

# via environment variable (server-side, shared token)
export GITHUB_TOKEN=ghp_your_token
uvicorn src.main:app --port 8000
```

---

## API reference

### `GET /v1/models`

Lists all models supported by the server.

```json
{
  "object": "list",
  "data": [
    { "id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "github-copilot" },
    ...
  ]
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions.  Supports `stream: true`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain async/await in Python."}],
    "stream": false
  }'
```

### `POST /v1/completions`

Legacy text-completion endpoint.  The prompt is forwarded as a user message to
the Copilot chat backend and the response is translated back to the
`text_completion` format.

---

## Content filtering

Every response is processed by a profanity filter before being returned.
Offensive words are replaced with `****` placeholders so the rest of the content
remains readable.

---

## Running tests

```bash
pytest
```

---

## Project structure

```
smart-api/
├── src/
│   ├── __init__.py
│   ├── main.py        # FastAPI application & route handlers
│   ├── copilot.py     # GitHub Copilot API client (auth + requests)
│   └── filter.py      # Profanity / content filter
├── tests/
│   ├── test_api.py    # API endpoint tests
│   └── test_filter.py # Content filter unit tests
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```
