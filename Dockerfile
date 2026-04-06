# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim

LABEL maintainer="KVBridge Team"
LABEL description="Cache-Aware OpenAI Compatible Proxy"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY kvbridge/ ./kvbridge/

# Non-root user
RUN useradd --create-home kvbridge
USER kvbridge

EXPOSE 8001

ENV KVBRIDGE_HOST=0.0.0.0
ENV KVBRIDGE_PORT=8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["python", "-m", "kvbridge.main"]
