# KVBridge

**Cache-Aware OpenAI Compatible Proxy for Agent Inference**

KVBridge is a lightweight, Cache-Aware OpenAI compatible proxy layer designed to solve the problem of **extremely low Prefix Cache Hit Rates** when autonomous Agents (like OpenCode, OpenHands, etc.) interact with inference engines such as vLLM, SGLang, and MindIE.

## ✨ Core Features

- **Transparent Proxy** — Fully compatible with the OpenAI `/v1/chat/completions` API. Agents only need a single line of configuration change.
- **Automatic Delta Compression** — Transforms the Agent's full-context resend into incremental delta payload transmission, preserving prefix stability.
- **Multi-Backend Support** — Seamlessly switch between vLLM, SGLang, and MindIE-Ascend backends.
- **Cache-Aware Protocol** — Exposes `/cache/status` and `/cache/append` endpoints to allow Agents to proactively participate in cache management.
- **Redis Persistence** — Supports Redis persistence for the Session prefix registry (with an in-memory fallback).
- **Prometheus Monitoring** — Real-time exposure of metrics like `prefix_hit_rate`, `delta_ratio`, and `latency`.

## 📊 Performance Expectations

| Metric | Without KVBridge | With KVBridge |
|--------|------------------|---------------|
| Prefix Cache Hit Rate | 5–15% | **80–95%** |
| Single Session Token Cost | Baseline | **Reduced 4–8×** |
| TTFT (Time To First Token) | Baseline | **Reduced 2–5×** |

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env to set the backend type and URL

# 2. Start services
docker compose up -d

# 3. Verify health
curl http://localhost:8001/health
```

### Option 2: Local Development

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Configure
cp .env.example .env

# 3. Start proxy
python -m kvbridge.main
```

### OpenCode Configuration

Simply modify your provider configuration to route through KVBridge:

```json
{
  "provider": {
    "kvbridge": {
      "baseURL": "http://localhost:8001/v1",
      "apiKey": "sk-no-key-required"
    }
  },
  "model": "kvbridge/qwen3-coder"
}
```

## ⚙️ Configuration Reference

All configuration is managed via environment variables (with a `KVBRIDGE_` prefix) or via the `.env` file:

| Environment Variable | Default Value | Description |
|----------------------|---------------|-------------|
| `KVBRIDGE_HOST` | `0.0.0.0` | Listening address |
| `KVBRIDGE_PORT` | `8001` | Listening port |
| `KVBRIDGE_BACKEND` | `vllm` | Backend engine: `vllm`, `sglang`, or `mindie` |
| `KVBRIDGE_BACKEND_URL` | `http://localhost:8000` | URL of the backend inference service |
| `KVBRIDGE_MODEL_NAME` | `qwen3-coder` | The model name exposed to clients |
| `KVBRIDGE_REDIS_URL` | *(empty)* | Redis URL. Leaves empty to use in-memory state |
| `KVBRIDGE_SESSION_TTL` | `3600` | Session expiration time (seconds) |
| `KVBRIDGE_MAX_CONTEXT_ROUNDS` | `3` | Number of recent conversation rounds to retain |
| `KVBRIDGE_ENABLE_DELTA` | `true` | Whether to enable Delta Compression |
| `KVBRIDGE_METRICS_ENABLED` | `true` | Whether to expose Prometheus metrics |

## 📡 API Documentation

### OpenAI Compatible Endpoints

```
POST /v1/chat/completions   # Fully compatible OpenAI Chat API
GET  /v1/models             # List available models
GET  /health                # Health check
```

### Cache-Aware Protocol

**Query Cache Status:**

```bash
curl "http://localhost:8001/cache/status?session_id=my-session"
```

```json
{
  "session_id": "my-session",
  "prefix_id": "kvbridge:my-session:abc12345",
  "hit_rate": 0.92,
  "prefix_messages_count": 5,
  "prefix_token_estimate": 2048,
  "max_delta_tokens": 8192,
  "last_updated": "2026-04-06T22:00:00+00:00"
}
```

**Append Delta Messages:**

```bash
curl -X POST http://localhost:8001/cache/append \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "prefix_id": "kvbridge:my-session:abc12345",
    "delta_messages": [
      {"role": "user", "content": "New follow-up question"}
    ]
  }'
```

### Response Headers

Each `/v1/chat/completions` response contains KVBridge diagnostic headers:

| Header | Description |
|--------|-------------|
| `X-KVBridge-Session` | The Session ID used for the request |
| `X-KVBridge-Prefix-Hit` | Whether the prefix cache was hit (`true`/`false`) |
| `X-KVBridge-Delta-Ratio` | Delta compression ratio (lower is better) |

## 🏗️ Architecture

```
OpenCode / OpenHands / AI Agents
        ↓ (Standard OpenAI /v1/chat/completions)
   [ KVBridge (FastAPI Proxy) ]
        ├── Session Tracker + Delta Compressor
        ├── /cache/status + /cache/append
        ├── Backend Router (vLLM / SGLang / MindIE)
        └── Prometheus Metrics (/metrics)
                ↓
   [ Inference Backend (vLLM / SGLang / MindIE) ]
         ↑
   Prefix Cache / RadixAttention / HiCache
```

## 🧪 Testing

```bash
pip install pytest pytest-asyncio
pytest -v
```

## 📁 Project Structure

```
kvbridge/
├── kvbridge/
│   ├── __init__.py            # Version info
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # pydantic-settings config
│   ├── session.py             # SessionTracker + DeltaCompressor
│   ├── cache_protocol.py      # /cache/status + /cache/append
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract interface + Backend factory
│   │   ├── vllm.py            # vLLM backend
│   │   ├── sglang.py          # SGLang backend
│   │   └── mindie.py          # MindIE-Ascend backend
│   ├── metrics.py             # Prometheus metrics definition
│   └── utils.py               # Utilities
├── tests/
│   ├── test_session.py        # Session/Delta unit tests
│   ├── test_api.py            # API integration tests
│   └── test_utils.py          # Utils unit tests
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

## 📜 License

MIT