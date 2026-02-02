"""
Pipeline Monitoring

Monitors pipeline health and sends alerts:
1. Source health tracking (failures, latency)
2. Data gap monitoring
3. Quality alerts
4. Telegram notifications
"""

import os
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path

import requests

import structlog

from data.pipeline.config import MonitoringConfig

logger = structlog.get_logger()


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SourceHealth(str, Enum):
    """Source health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class SourceStatus:
    """Status of a data source."""

    name: str
    health: SourceHealth = SourceHealth.UNKNOWN
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)

    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.last_success = datetime.now(timezone.utc)
        self.consecutive_failures = 0
        self.total_requests += 1
        self.health = SourceHealth.HEALTHY

        # Update latency (keep last 100 samples)
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
        self.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    def record_failure(self, error: str):
        """Record a failed request."""
        self.last_failure = datetime.now(timezone.utc)
        self.last_error = error
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1

        # Update health
        if self.consecutive_failures >= 5:
            self.health = SourceHealth.DOWN
        elif self.consecutive_failures >= 2:
            self.health = SourceHealth.DEGRADED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "health": self.health.value,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": (self.total_requests - self.total_failures) / max(1, self.total_requests),
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class Alert:
    """An alert event."""

    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "symbol": self.symbol,
            "metadata": self.metadata,
        }


class TelegramNotifier:
    """Sends alerts to Telegram."""

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token.
            chat_id: Telegram chat ID.
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        self._last_alert_time: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Minimum seconds between same alerts

    def send(self, alert: Alert) -> bool:
        """
        Send alert to Telegram.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        # Rate limiting - don't spam same alert
        alert_key = f"{alert.level}:{alert.title}"
        last_time = self._last_alert_time.get(alert_key)
        if last_time:
            elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
            if elapsed < self._rate_limit_seconds:
                return False

        # Format message
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "🔴",
            AlertLevel.CRITICAL: "🚨",
        }.get(alert.level, "📢")

        message = f"{emoji} *{alert.title}*\n\n{alert.message}"

        if alert.source:
            message += f"\n\nSource: `{alert.source}`"

        if alert.symbol:
            message += f"\nSymbol: `{alert.symbol}`"

        message += f"\n\n_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_"

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            response = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            response.raise_for_status()

            self._last_alert_time[alert_key] = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False


class PipelineMonitor:
    """
    Monitors pipeline health and generates alerts.

    Features:
    - Source health tracking
    - Gap detection monitoring
    - Quality score monitoring
    - Telegram alerts
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize monitor.

        Args:
            config: Monitoring configuration.
        """
        self.config = config or MonitoringConfig.from_env()

        # Source status tracking
        self.sources: Dict[str, SourceStatus] = {}

        # Alert history
        self.alerts: List[Alert] = []
        self._max_alerts = 1000

        # Telegram notifier
        self.telegram = TelegramNotifier(
            self.config.telegram_bot_token,
            self.config.telegram_chat_id,
        )

        # Gap tracking
        self.detected_gaps: List[Dict[str, Any]] = []

        # Quality tracking
        self.quality_history: Dict[str, List[float]] = {}

        # Background health checker
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_background_monitoring(self):
        """Start background health monitoring thread."""
        if self._health_check_thread is not None:
            return

        self._stop_event.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
        )
        self._health_check_thread.start()
        logger.info("Started background monitoring")

    def stop_background_monitoring(self):
        """Stop background monitoring thread."""
        if self._health_check_thread is None:
            return

        self._stop_event.set()
        self._health_check_thread.join(timeout=5)
        self._health_check_thread = None
        logger.info("Stopped background monitoring")

    def _health_check_loop(self):
        """Background health check loop."""
        while not self._stop_event.is_set():
            try:
                self._check_source_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._stop_event.wait(self.config.health_check_interval_seconds)

    def _check_source_health(self):
        """Check health of all sources."""
        now = datetime.now(timezone.utc)
        threshold = timedelta(minutes=self.config.source_failure_alert_minutes)

        for name, status in self.sources.items():
            # Check if source has been down too long
            if status.last_failure and status.health == SourceHealth.DOWN:
                time_since_success = now - (status.last_success or datetime.min.replace(tzinfo=timezone.utc))

                if time_since_success > threshold:
                    self._create_alert(
                        level=AlertLevel.ERROR,
                        title=f"Data Source Down: {name}",
                        message=f"Source {name} has been down for {time_since_success}.\n"
                                f"Last error: {status.last_error}",
                        source=name,
                    )

    def record_request_success(
        self,
        source: str,
        latency_ms: float,
        symbol: Optional[str] = None,
    ):
        """Record a successful request."""
        if source not in self.sources:
            self.sources[source] = SourceStatus(name=source)

        status = self.sources[source]

        # Check if recovering from failure
        was_down = status.health == SourceHealth.DOWN

        status.record_success(latency_ms)

        if was_down:
            self._create_alert(
                level=AlertLevel.INFO,
                title=f"Data Source Recovered: {source}",
                message=f"Source {source} is back online.",
                source=source,
            )

    def record_request_failure(
        self,
        source: str,
        error: str,
        symbol: Optional[str] = None,
    ):
        """Record a failed request."""
        if source not in self.sources:
            self.sources[source] = SourceStatus(name=source)

        status = self.sources[source]
        status.record_failure(error)

        # Alert if newly down
        if status.consecutive_failures == 3:
            self._create_alert(
                level=AlertLevel.WARNING,
                title=f"Data Source Degraded: {source}",
                message=f"Source {source} has failed {status.consecutive_failures} consecutive times.\n"
                        f"Error: {error}",
                source=source,
                symbol=symbol,
            )
        elif status.consecutive_failures == 5:
            self._create_alert(
                level=AlertLevel.ERROR,
                title=f"Data Source Down: {source}",
                message=f"Source {source} is down after {status.consecutive_failures} failures.\n"
                        f"Error: {error}",
                source=source,
                symbol=symbol,
            )

    def record_gap(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        bars_missing: int,
    ):
        """Record a detected data gap."""
        gap = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bars_missing": bars_missing,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
        self.detected_gaps.append(gap)

        # Alert if significant gap
        if bars_missing >= self.config.gap_alert_min_bars:
            self._create_alert(
                level=AlertLevel.WARNING,
                title=f"Data Gap Detected: {symbol}",
                message=f"Gap of {bars_missing} bars detected in {symbol} {timeframe}.\n"
                        f"From: {start.isoformat()}\n"
                        f"To: {end.isoformat()}",
                symbol=symbol,
            )

    def record_quality(
        self,
        symbol: str,
        timeframe: str,
        quality_score: float,
    ):
        """Record quality score."""
        key = f"{symbol}_{timeframe}"

        if key not in self.quality_history:
            self.quality_history[key] = []

        self.quality_history[key].append(quality_score)

        # Keep last 100 scores
        if len(self.quality_history[key]) > 100:
            self.quality_history[key] = self.quality_history[key][-100:]

        # Alert if below threshold
        if quality_score < self.config.quality_alert_threshold:
            self._create_alert(
                level=AlertLevel.WARNING,
                title=f"Low Data Quality: {symbol}",
                message=f"Quality score for {symbol} {timeframe} is {quality_score:.2f} "
                        f"(threshold: {self.config.quality_alert_threshold})",
                symbol=symbol,
            )

    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: Optional[str] = None,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create and dispatch an alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            symbol=symbol,
            metadata=metadata or {},
        )

        # Store alert
        self.alerts.append(alert)
        if len(self.alerts) > self._max_alerts:
            self.alerts = self.alerts[-self._max_alerts:]

        # Log
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)

        log_method(title, **alert.to_dict())

        # Send to Telegram
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL, AlertLevel.WARNING]:
            self.telegram.send(alert)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "sources": {
                name: status.to_dict()
                for name, status in self.sources.items()
            },
            "summary": {
                "total_sources": len(self.sources),
                "healthy_sources": sum(
                    1 for s in self.sources.values() if s.health == SourceHealth.HEALTHY
                ),
                "degraded_sources": sum(
                    1 for s in self.sources.values() if s.health == SourceHealth.DEGRADED
                ),
                "down_sources": sum(
                    1 for s in self.sources.values() if s.health == SourceHealth.DOWN
                ),
            },
            "gaps": {
                "total_detected": len(self.detected_gaps),
                "recent": self.detected_gaps[-10:],
            },
            "alerts": {
                "total": len(self.alerts),
                "recent": [a.to_dict() for a in self.alerts[-10:]],
            },
            "quality": {
                key: {
                    "latest": scores[-1] if scores else None,
                    "avg": sum(scores) / len(scores) if scores else None,
                    "min": min(scores) if scores else None,
                }
                for key, scores in self.quality_history.items()
            },
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get concise health summary."""
        healthy = sum(1 for s in self.sources.values() if s.health == SourceHealth.HEALTHY)
        total = len(self.sources)

        overall = "HEALTHY" if healthy == total else "DEGRADED" if healthy > 0 else "DOWN"

        return {
            "overall": overall,
            "sources": f"{healthy}/{total} healthy",
            "gaps_24h": sum(
                1 for g in self.detected_gaps
                if datetime.fromisoformat(g["detected_at"]) > datetime.now(timezone.utc) - timedelta(days=1)
            ),
            "alerts_24h": sum(
                1 for a in self.alerts
                if a.timestamp > datetime.now(timezone.utc) - timedelta(days=1)
            ),
        }
