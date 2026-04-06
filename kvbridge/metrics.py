"""Prometheus metrics collection."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge


# ----- Counters -----
REQUEST_TOTAL = Counter(
    "kvbridge_request_total",
    "Total requests received",
    ["method", "endpoint", "status"],
)

PREFIX_HIT_TOTAL = Counter(
    "kvbridge_prefix_hit_total",
    "Requests where prefix cache was reused",
)

PREFIX_MISS_TOTAL = Counter(
    "kvbridge_prefix_miss_total",
    "Requests where no prefix cache existed",
)

# ----- Histograms -----
DELTA_RATIO = Histogram(
    "kvbridge_delta_ratio",
    "Ratio of delta tokens to total tokens (lower = better compression)",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
)

BACKEND_LATENCY = Histogram(
    "kvbridge_backend_latency_seconds",
    "Latency of backend inference call",
    ["backend"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)

PROXY_LATENCY = Histogram(
    "kvbridge_proxy_latency_seconds",
    "Total proxy processing latency (excluding backend)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

# ----- Gauges -----
ACTIVE_SESSIONS = Gauge(
    "kvbridge_active_sessions",
    "Number of active sessions in tracker",
)

PREFIX_HIT_RATE = Gauge(
    "kvbridge_prefix_hit_rate",
    "Rolling prefix cache hit rate",
)
