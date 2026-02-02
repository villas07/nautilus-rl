"""
Metrics Collection for Prometheus/Grafana

Exposes trading metrics for monitoring dashboards.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

import structlog

logger = structlog.get_logger()

# Try to import prometheus client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed")


@dataclass
class MetricsConfig:
    """Metrics configuration."""

    enabled: bool = True
    prefix: str = "nautilus_agents"
    port: int = 8000


class MetricsCollector:
    """
    Collects and exposes metrics for Prometheus.

    Metrics:
    - Trading: trades, PnL, positions
    - System: CPU, memory, latency
    - Models: predictions, voting results
    - Errors: failures, retries
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
    ):
        """Initialize metrics collector."""
        self.config = config or MetricsConfig()
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

        if PROMETHEUS_AVAILABLE and self.config.enabled:
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        prefix = self.config.prefix

        # Trading metrics
        self.trades_total = Counter(
            f"{prefix}_trades_total",
            "Total number of trades",
            ["instrument", "side"],
            registry=self._registry,
        )

        self.trade_pnl = Summary(
            f"{prefix}_trade_pnl",
            "Trade PnL distribution",
            ["instrument"],
            registry=self._registry,
        )

        self.position_value = Gauge(
            f"{prefix}_position_value",
            "Current position value",
            ["instrument"],
            registry=self._registry,
        )

        self.daily_pnl = Gauge(
            f"{prefix}_daily_pnl",
            "Daily PnL",
            registry=self._registry,
        )

        self.equity = Gauge(
            f"{prefix}_equity",
            "Current equity",
            registry=self._registry,
        )

        self.drawdown = Gauge(
            f"{prefix}_drawdown",
            "Current drawdown",
            registry=self._registry,
        )

        # Model metrics
        self.predictions_total = Counter(
            f"{prefix}_predictions_total",
            "Total model predictions",
            ["agent_id", "signal"],
            registry=self._registry,
        )

        self.prediction_latency = Histogram(
            f"{prefix}_prediction_latency_seconds",
            "Model prediction latency",
            ["agent_id"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self._registry,
        )

        self.voting_confidence = Gauge(
            f"{prefix}_voting_confidence",
            "Voting system confidence",
            ["instrument"],
            registry=self._registry,
        )

        self.active_models = Gauge(
            f"{prefix}_active_models",
            "Number of active models",
            registry=self._registry,
        )

        # System metrics
        self.cpu_usage = Gauge(
            f"{prefix}_cpu_usage_percent",
            "CPU usage percentage",
            registry=self._registry,
        )

        self.memory_usage = Gauge(
            f"{prefix}_memory_usage_percent",
            "Memory usage percentage",
            registry=self._registry,
        )

        self.order_latency = Histogram(
            f"{prefix}_order_latency_seconds",
            "Order execution latency",
            ["instrument", "order_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self._registry,
        )

        # Error metrics
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors",
            ["type"],
            registry=self._registry,
        )

        self.circuit_breaker_trips = Counter(
            f"{prefix}_circuit_breaker_trips_total",
            "Circuit breaker trip count",
            registry=self._registry,
        )

        # Info metric
        self.info = Info(
            f"{prefix}_info",
            "Trading system information",
            registry=self._registry,
        )
        self.info.info({
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production"),
        })

    # ========================================================================
    # Recording Methods
    # ========================================================================

    def record_trade(
        self,
        instrument: str,
        side: str,
        pnl: float,
    ) -> None:
        """Record a trade."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.trades_total.labels(instrument=instrument, side=side).inc()
        self.trade_pnl.labels(instrument=instrument).observe(pnl)

    def record_position(
        self,
        instrument: str,
        value: float,
    ) -> None:
        """Record position value."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.position_value.labels(instrument=instrument).set(value)

    def record_pnl(
        self,
        daily: float,
        equity: float,
        drawdown: float,
    ) -> None:
        """Record PnL metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.daily_pnl.set(daily)
        self.equity.set(equity)
        self.drawdown.set(drawdown)

    def record_prediction(
        self,
        agent_id: str,
        signal: int,
        latency: float,
    ) -> None:
        """Record a model prediction."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        signal_str = {1: "buy", -1: "sell", 0: "hold"}.get(signal, "unknown")
        self.predictions_total.labels(agent_id=agent_id, signal=signal_str).inc()
        self.prediction_latency.labels(agent_id=agent_id).observe(latency)

    def record_voting(
        self,
        instrument: str,
        confidence: float,
        active_models: int,
    ) -> None:
        """Record voting results."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.voting_confidence.labels(instrument=instrument).set(confidence)
        self.active_models.set(active_models)

    def record_order_latency(
        self,
        instrument: str,
        order_type: str,
        latency: float,
    ) -> None:
        """Record order execution latency."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.order_latency.labels(
            instrument=instrument,
            order_type=order_type,
        ).observe(latency)

    def record_system_metrics(
        self,
        cpu: float,
        memory: float,
    ) -> None:
        """Record system resource metrics."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.cpu_usage.set(cpu)
        self.memory_usage.set(memory)

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.errors_total.labels(type=error_type).inc()

    def record_circuit_breaker(self) -> None:
        """Record circuit breaker trip."""
        if not PROMETHEUS_AVAILABLE or not self.config.enabled:
            return

        self.circuit_breaker_trips.inc()

    # ========================================================================
    # Export Methods
    # ========================================================================

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return b"# prometheus_client not installed\n"

        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        if PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# Singleton instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create the metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


# FastAPI integration
def create_metrics_app():
    """Create FastAPI app for metrics endpoint."""
    from fastapi import FastAPI, Response

    app = FastAPI(title="Nautilus Agents Metrics")
    metrics = get_metrics()

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(
            content=metrics.get_metrics(),
            media_type=metrics.get_content_type(),
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app
