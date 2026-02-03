"""
Health Check System

Monitors system health and sends alerts when issues are detected.

Requires: psutil
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    raise ImportError(
        "psutil is required for health_checks. "
        "Install with: pip install psutil"
    )

import structlog

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class HealthCheckConfig:
    """Health check configuration."""

    check_interval_seconds: int = 300  # 5 minutes
    timeout_seconds: int = 30
    max_consecutive_failures: int = 3
    alert_on_degraded: bool = True


class HealthChecker:
    """
    System health monitoring.

    Checks:
    - IBKR Gateway connection
    - Binance connection
    - Database connection
    - Model availability
    - System resources
    - Trading activity
    """

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize health checker."""
        self.config = config or HealthCheckConfig()
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._failure_counts: Dict[str, int] = {}

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("models_loaded", self._check_models)
        self.register_check("database", self._check_database)
        self.register_check("ibkr_gateway", self._check_ibkr)
        self.register_check("binance_api", self._check_binance)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_func
        self._failure_counts[name] = 0

    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}

        for name, check_func in self._checks.items():
            start = datetime.now()

            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(
                        check_func(),
                        timeout=self.config.timeout_seconds,
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, check_func
                    )

                result.duration_ms = (datetime.now() - start).total_seconds() * 1000

            except asyncio.TimeoutError:
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check timed out",
                    timestamp=datetime.now(),
                )
            except Exception as e:
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    timestamp=datetime.now(),
                )

            results[name] = result
            self._results[name] = result

            # Track failures
            if result.status == HealthStatus.UNHEALTHY:
                self._failure_counts[name] += 1
            else:
                self._failure_counts[name] = 0

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self._results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            "status": self.get_overall_status().value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                }
                for name, result in self._results.items()
            },
            "failure_counts": self._failure_counts,
        }

    # ========================================================================
    # Individual Checks
    # ========================================================================

    def _check_system_resources(self) -> HealthCheckResult:
        """Check CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 ** 3),
        }

        if cpu_percent > 90 or memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High resource usage: CPU={cpu_percent}%, Memory={memory.percent}%"
        elif cpu_percent > 70 or memory.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Elevated resource usage: CPU={cpu_percent}%, Memory={memory.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resources OK: CPU={cpu_percent}%, Memory={memory.percent}%"

        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space."""
        disk = psutil.disk_usage("/")
        free_gb = disk.free / (1024 ** 3)
        used_percent = disk.percent

        details = {
            "free_gb": free_gb,
            "used_percent": used_percent,
        }

        if free_gb < 1 or used_percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Critical disk space: {free_gb:.1f}GB free"
        elif free_gb < 5 or used_percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Low disk space: {free_gb:.1f}GB free"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk OK: {free_gb:.1f}GB free"

        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )

    def _check_models(self) -> HealthCheckResult:
        """Check if models are loaded."""
        try:
            from live.model_loader import ModelLoader

            loader = ModelLoader()
            agents = loader.get_validated_agents()

            if len(agents) >= 10:
                status = HealthStatus.HEALTHY
                message = f"{len(agents)} models available"
            elif len(agents) > 0:
                status = HealthStatus.DEGRADED
                message = f"Only {len(agents)} models available"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No models available"

            details = {"model_count": len(agents)}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Model check failed: {e}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="models_loaded",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )

    def _check_database(self) -> HealthCheckResult:
        """Check database connection."""
        try:
            from data.adapters.timescale_adapter import TimescaleAdapter

            adapter = TimescaleAdapter()
            symbols = adapter.get_available_symbols("1h")
            adapter.close()

            if len(symbols) > 0:
                status = HealthStatus.HEALTHY
                message = f"Database OK: {len(symbols)} symbols"
            else:
                status = HealthStatus.DEGRADED
                message = "Database connected but no data"

            details = {"symbol_count": len(symbols)}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database connection failed: {e}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="database",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )

    def _check_ibkr(self) -> HealthCheckResult:
        """Check IBKR Gateway connection."""
        import socket

        host = os.getenv("IBKR_HOST", "localhost")
        port = int(os.getenv("IBKR_PORT", "7497"))

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                status = HealthStatus.HEALTHY
                message = f"IBKR Gateway reachable at {host}:{port}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"IBKR Gateway not reachable at {host}:{port}"

            details = {"host": host, "port": port}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"IBKR check failed: {e}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="ibkr_gateway",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )

    def _check_binance(self) -> HealthCheckResult:
        """Check Binance API connectivity."""
        import socket

        try:
            # Simple DNS lookup and TCP connect
            host = "fapi.binance.com"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((socket.gethostbyname(host), 443))
            sock.close()

            if result == 0:
                status = HealthStatus.HEALTHY
                message = "Binance API reachable"
            else:
                status = HealthStatus.DEGRADED
                message = "Binance API connection issues"

            details = {"host": host}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Binance check failed: {e}"
            details = {"error": str(e)}

        return HealthCheckResult(
            name="binance_api",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
        )


async def run_health_checks() -> Dict[str, Any]:
    """Convenience function to run all health checks."""
    checker = HealthChecker()
    await checker.run_all_checks()
    return checker.get_summary()


# Background health check loop
async def health_check_loop(
    interval_seconds: int = 300,
    callback: Optional[Callable] = None,
):
    """Run health checks in a loop."""
    checker = HealthChecker()

    while True:
        try:
            await checker.run_all_checks()
            summary = checker.get_summary()

            logger.info(
                "Health check completed",
                status=summary["status"],
            )

            if callback:
                await callback(summary)

            # Alert on issues
            if summary["status"] in ["unhealthy", "degraded"]:
                from live.telegram_alerts import get_alerts, AlertLevel

                alerts = get_alerts()
                await alerts.send_message(
                    f"⚠️ System Health: {summary['status'].upper()}\n\n"
                    + "\n".join(
                        f"• {name}: {data['message']}"
                        for name, data in summary["checks"].items()
                        if data["status"] != "healthy"
                    ),
                    AlertLevel.WARNING,
                )

        except Exception as e:
            logger.error(f"Health check loop error: {e}")

        await asyncio.sleep(interval_seconds)
