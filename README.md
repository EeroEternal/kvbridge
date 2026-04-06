# KVBridge

**Cache-Aware OpenAI Compatible Proxy for Agent Inference**

KVBridge 是一个轻量级、Cache-Aware 的 OpenAI 兼容代理层，专为解决 Agent（OpenCode / OpenHands 等）在 vLLM / SGLang / MindIE 等推理引擎上 **Prefix Cache Hit Rate 极低** 的问题而设计。

## ✨ 核心特性

- **透明代理** — 完全兼容 OpenAI `/v1/chat/completions` API，Agent 只需改一行配置
- **自动 Delta 压缩** — 将 Agent 的全量重发改为增量传输，保护 prefix 稳定性
- **三后端支持** — vLLM / SGLang / MindIE-Ascend 无缝切换
- **Cache-Aware 协议** — `/cache/status` + `/cache/append` 让 Agent 主动参与缓存管理
- **Redis 持久化** — Session prefix registry 支持 Redis 持久化（可选内存 fallback）
- **Prometheus 监控** — prefix_hit_rate、delta_ratio、latency 等指标实时暴露

## 📊 性能预期

| 指标 | 无 KVBridge | 有 KVBridge |
|------|-------------|-------------|
| Prefix Cache Hit Rate | 5–15% | **80–95%** |
| 单 Session Token 消耗 | 基准 | **降低 4–8×** |
| TTFT (首 Token 延迟) | 基准 | **降低 2–5×** |

## 🚀 快速启动

### 方式一：Docker Compose（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，设置后端类型和地址

# 2. 启动服务
docker compose up -d

# 3. 验证
curl http://localhost:8001/health
```

### 方式二：本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp .env.example .env

# 3. 启动
python -m kvbridge.main
```

### OpenCode 配置

只需修改 provider 配置，即可接入 KVBridge：

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

## ⚙️ 配置说明

所有配置通过环境变量（`KVBRIDGE_` 前缀）或 `.env` 文件：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `KVBRIDGE_HOST` | `0.0.0.0` | 监听地址 |
| `KVBRIDGE_PORT` | `8001` | 监听端口 |
| `KVBRIDGE_BACKEND` | `vllm` | 后端类型：`vllm` / `sglang` / `mindie` |
| `KVBRIDGE_BACKEND_URL` | `http://localhost:8000` | 后端推理服务地址 |
| `KVBRIDGE_MODEL_NAME` | `qwen3-coder` | 对外暴露的模型名称 |
| `KVBRIDGE_REDIS_URL` | *(空)* | Redis 地址，空则使用内存 |
| `KVBRIDGE_SESSION_TTL` | `3600` | Session 过期时间（秒） |
| `KVBRIDGE_MAX_CONTEXT_ROUNDS` | `3` | 保留的最近对话轮数 |
| `KVBRIDGE_ENABLE_DELTA` | `true` | 是否启用 Delta 压缩 |
| `KVBRIDGE_METRICS_ENABLED` | `true` | 是否启用 Prometheus 指标 |

## 📡 API 文档

### OpenAI 兼容接口

```
POST /v1/chat/completions   # 完全兼容 OpenAI Chat API
GET  /v1/models              # 模型列表
GET  /health                 # 健康检查
```

### Cache-Aware 协议

**查询缓存状态：**

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

**追加增量消息：**

```bash
curl -X POST http://localhost:8001/cache/append \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "prefix_id": "kvbridge:my-session:abc12345",
    "delta_messages": [
      {"role": "user", "content": "新的问题"}
    ]
  }'
```

### 响应头

每个 `/v1/chat/completions` 响应都包含 KVBridge 诊断头：

| Header | 说明 |
|--------|------|
| `X-KVBridge-Session` | 使用的 Session ID |
| `X-KVBridge-Prefix-Hit` | 是否命中前缀缓存 (`true`/`false`) |
| `X-KVBridge-Delta-Ratio` | Delta 压缩比 (越低越好) |

## 🏗️ 架构

```
OpenCode / OpenHands
        ↓ (标准 OpenAI /v1/chat/completions)
   [ KVBridge (FastAPI Proxy) ]
        ├── Session Tracker + Delta Compressor
        ├── /cache/status + /cache/append
        ├── Backend Router (vLLM / SGLang / MindIE)
        └── Prometheus Metrics (/metrics)
                ↓
   [ 推理后端 (vLLM / SGLang / MindIE) ]
         ↑
   Prefix Cache / RadixAttention / HiCache
```

## 🧪 测试

```bash
pip install pytest pytest-asyncio
pytest -v
```

## 📁 项目结构

```
kvbridge/
├── kvbridge/
│   ├── __init__.py            # 版本号
│   ├── main.py                # FastAPI 应用入口
│   ├── config.py              # pydantic-settings 配置
│   ├── session.py             # SessionTracker + DeltaCompressor
│   ├── cache_protocol.py      # /cache/status + /cache/append
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── base.py            # 抽象后端接口 + 工厂函数
│   │   ├── vllm.py            # vLLM 后端
│   │   ├── sglang.py          # SGLang 后端
│   │   └── mindie.py          # MindIE-Ascend 后端
│   ├── metrics.py             # Prometheus 指标定义
│   └── utils.py               # 工具函数
├── tests/
│   ├── test_session.py        # Session/Delta 单元测试
│   ├── test_api.py            # API 集成测试
│   └── test_utils.py          # 工具函数测试
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