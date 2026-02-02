"""
Autonomous Controller

Manages the trading system autonomously based on config/autonomous_config.yaml.
Only alerts on CRITICAL events. All other decisions are automatic.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CAUTION = "caution"
    CRITICAL = "critical"


class Action(Enum):
    NONE = "none"
    LOG_ONLY = "log_only"
    REDUCE_EXPOSURE = "reduce_exposure"
    PAUSE_TRADING = "pause_trading"
    SKIP_TRADE = "skip_trade"
    ALERT_TELEGRAM = "alert_telegram"


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker status."""
    drawdown_pct: float = 0.0
    daily_loss_usd: float = 0.0
    consecutive_losses: int = 0
    last_data_timestamp: Optional[datetime] = None
    is_paused: bool = False
    pause_reason: Optional[str] = None
    exposure_reduction_pct: float = 0.0


@dataclass
class TradingState:
    """Current trading state."""
    total_exposure_usd: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    daily_pnl_usd: float = 0.0
    active_models: int = 0
    last_signal_time: Dict[str, datetime] = field(default_factory=dict)


class AutonomousController:
    """
    Controls the trading system autonomously.

    Responsibilities:
    - Load and apply configuration
    - Monitor circuit breakers
    - Make position sizing decisions
    - Trigger alerts only when critical
    - Log all decisions for audit
    """

    def __init__(self, config_path: str = None):
        """Initialize controller."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "autonomous_config.yaml"

        self.config = self._load_config(config_path)
        self.circuit_state = CircuitBreakerState()
        self.trading_state = TradingState()
        self.telegram_notifier = None  # Inject later

        logger.info("AutonomousController initialized", config_path=str(config_path))

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        with open(path) as f:
            return yaml.safe_load(f)

    # =========================================================================
    # CIRCUIT BREAKERS
    # =========================================================================

    def check_circuit_breakers(self) -> tuple[AlertLevel, Action]:
        """
        Check all circuit breakers and return highest alert level and action.

        Returns:
            Tuple of (AlertLevel, Action) indicating system state.
        """
        checks = [
            self._check_drawdown(),
            self._check_daily_loss(),
            self._check_consecutive_losses(),
            self._check_data_staleness(),
        ]

        # Get highest severity
        max_level = AlertLevel.INFO
        action = Action.NONE

        for level, act, reason in checks:
            if level.value > max_level.value:
                max_level = level
                action = act
                if level == AlertLevel.CRITICAL:
                    self.circuit_state.is_paused = True
                    self.circuit_state.pause_reason = reason

        return max_level, action

    def _check_drawdown(self) -> tuple[AlertLevel, Action, str]:
        """Check drawdown circuit breaker."""
        cfg = self.config["circuit_breakers"]["drawdown"]
        dd = self.circuit_state.drawdown_pct

        if dd >= cfg["critical_pct"]:
            return AlertLevel.CRITICAL, Action.PAUSE_TRADING, f"Drawdown {dd:.1f}% >= {cfg['critical_pct']}%"
        elif dd >= cfg["caution_pct"]:
            self.circuit_state.exposure_reduction_pct = 50
            return AlertLevel.CAUTION, Action.REDUCE_EXPOSURE, f"Drawdown {dd:.1f}%"
        elif dd >= cfg["warning_pct"]:
            return AlertLevel.WARNING, Action.LOG_ONLY, f"Drawdown {dd:.1f}%"

        return AlertLevel.INFO, Action.NONE, ""

    def _check_daily_loss(self) -> tuple[AlertLevel, Action, str]:
        """Check daily loss circuit breaker."""
        cfg = self.config["circuit_breakers"]["daily_loss"]
        loss = abs(min(0, self.trading_state.daily_pnl_usd))

        if loss >= cfg["critical_usd"]:
            return AlertLevel.CRITICAL, Action.PAUSE_TRADING, f"Daily loss ${loss:.0f} >= ${cfg['critical_usd']}"
        elif loss >= cfg["caution_usd"]:
            self.circuit_state.exposure_reduction_pct = 50
            return AlertLevel.CAUTION, Action.REDUCE_EXPOSURE, f"Daily loss ${loss:.0f}"
        elif loss >= cfg["warning_usd"]:
            return AlertLevel.WARNING, Action.LOG_ONLY, f"Daily loss ${loss:.0f}"

        return AlertLevel.INFO, Action.NONE, ""

    def _check_consecutive_losses(self) -> tuple[AlertLevel, Action, str]:
        """Check consecutive losses circuit breaker."""
        cfg = self.config["circuit_breakers"]["consecutive_losses"]
        losses = self.circuit_state.consecutive_losses

        if losses >= cfg["critical_count"]:
            self.circuit_state.exposure_reduction_pct = 50
            return AlertLevel.CRITICAL, Action.REDUCE_EXPOSURE, f"{losses} consecutive losses"
        elif losses >= cfg["caution_count"]:
            self.circuit_state.exposure_reduction_pct = 50
            return AlertLevel.CAUTION, Action.REDUCE_EXPOSURE, f"{losses} consecutive losses"
        elif losses >= cfg["warning_count"]:
            return AlertLevel.WARNING, Action.LOG_ONLY, f"{losses} consecutive losses"

        return AlertLevel.INFO, Action.NONE, ""

    def _check_data_staleness(self) -> tuple[AlertLevel, Action, str]:
        """Check data staleness circuit breaker."""
        cfg = self.config["circuit_breakers"]["data_staleness"]

        if self.circuit_state.last_data_timestamp is None:
            return AlertLevel.INFO, Action.NONE, ""

        staleness = (datetime.now() - self.circuit_state.last_data_timestamp).total_seconds()

        if staleness >= cfg["critical_seconds"]:
            return AlertLevel.CRITICAL, Action.PAUSE_TRADING, f"Data stale for {staleness:.0f}s"
        elif staleness >= cfg["caution_seconds"]:
            return AlertLevel.CAUTION, Action.LOG_ONLY, f"Data stale for {staleness:.0f}s"
        elif staleness >= cfg["warning_seconds"]:
            return AlertLevel.WARNING, Action.LOG_ONLY, f"Data stale for {staleness:.0f}s"

        return AlertLevel.INFO, Action.NONE, ""

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        confidence: float,
        signal_direction: int,  # 1 = long, -1 = short
    ) -> float:
        """
        Calculate position size based on confidence and config.

        Returns:
            Position size in USD. 0 means skip trade.
        """
        cfg = self.config["position_sizing"]

        # Check minimum confidence
        if confidence < cfg["confidence_scaling"]["min_confidence"]:
            logger.info("Trade skipped - low confidence",
                       symbol=symbol, confidence=confidence)
            return 0.0

        # Check if paused
        if self.circuit_state.is_paused:
            logger.info("Trade skipped - system paused",
                       symbol=symbol, reason=self.circuit_state.pause_reason)
            return 0.0

        # Base size
        base = cfg["base_position_usd"]

        # Scale by confidence
        scaling = cfg["confidence_scaling"]
        if confidence >= scaling["high_confidence"]:
            multiplier = 1.5
        elif confidence >= scaling["medium_confidence"]:
            multiplier = 1.0
        elif confidence >= scaling["low_confidence"]:
            multiplier = 0.5
        else:
            multiplier = 0.0

        size = base * multiplier

        # Apply exposure reduction if circuit breaker active
        if self.circuit_state.exposure_reduction_pct > 0:
            size *= (1 - self.circuit_state.exposure_reduction_pct / 100)

        # Check per-symbol limit
        current_position = self.trading_state.positions.get(symbol, 0)
        max_per_symbol = cfg["max_position_per_symbol_usd"]
        if abs(current_position) + size > max_per_symbol:
            size = max(0, max_per_symbol - abs(current_position))

        # Check total exposure limit
        max_total = cfg["max_total_exposure_usd"]
        if self.trading_state.total_exposure_usd + size > max_total:
            size = max(0, max_total - self.trading_state.total_exposure_usd)

        logger.info("Position size calculated",
                   symbol=symbol,
                   confidence=confidence,
                   size_usd=size,
                   multiplier=multiplier)

        return size

    # =========================================================================
    # VOTING & SIGNALS
    # =========================================================================

    def evaluate_voting_signal(
        self,
        symbol: str,
        model_signals: Dict[str, int],  # model_id -> signal (-1, 0, 1)
        model_weights: Dict[str, float],  # model_id -> weight (e.g., Sharpe)
    ) -> tuple[int, float]:
        """
        Evaluate voting signal from multiple models.

        Returns:
            Tuple of (signal, confidence).
            signal: -1 (sell), 0 (hold), 1 (buy)
            confidence: 0.0 to 1.0
        """
        cfg = self.config["voting"]

        # Check minimum models
        if len(model_signals) < cfg["min_models_required"]:
            logger.warning("Not enough models for voting",
                          required=cfg["min_models_required"],
                          available=len(model_signals))
            return 0, 0.0

        # Check cooldown
        last_signal = self.trading_state.last_signal_time.get(symbol)
        if last_signal:
            cooldown = timedelta(minutes=cfg["signal_cooldown_minutes"])
            if datetime.now() - last_signal < cooldown:
                logger.info("Signal in cooldown", symbol=symbol)
                return 0, 0.0

        # Weighted voting
        if cfg["aggregation_method"] == "weighted_confidence":
            weighted_sum = 0.0
            total_weight = 0.0

            for model_id, signal in model_signals.items():
                weight = model_weights.get(model_id, 1.0)
                weighted_sum += signal * weight
                total_weight += weight

            if total_weight == 0:
                return 0, 0.0

            avg_signal = weighted_sum / total_weight

            # Determine direction and confidence
            if avg_signal > 0.1:
                direction = 1
                confidence = min(1.0, avg_signal)
            elif avg_signal < -0.1:
                direction = -1
                confidence = min(1.0, abs(avg_signal))
            else:
                direction = 0
                confidence = 0.0

        else:  # majority voting
            buy_votes = sum(1 for s in model_signals.values() if s > 0)
            sell_votes = sum(1 for s in model_signals.values() if s < 0)
            total = len(model_signals)

            if buy_votes > sell_votes:
                direction = 1
                confidence = buy_votes / total
            elif sell_votes > buy_votes:
                direction = -1
                confidence = sell_votes / total
            else:
                direction = 0
                confidence = 0.0

        # Check minimum agreement
        if confidence < cfg["min_agreement_pct"] / 100:
            logger.info("Voting confidence too low",
                       symbol=symbol, confidence=confidence)
            return 0, confidence

        # Update last signal time
        self.trading_state.last_signal_time[symbol] = datetime.now()

        return direction, confidence

    # =========================================================================
    # ALERTS
    # =========================================================================

    def send_alert(self, level: AlertLevel, message: str, data: Dict = None):
        """
        Send alert based on level.
        Only CRITICAL goes to Telegram.
        """
        log_data = {"level": level.value, "message": message, **(data or {})}

        if level == AlertLevel.CRITICAL:
            logger.error("CRITICAL ALERT", **log_data)
            self._send_telegram(f"🚨 CRITICAL: {message}")
        elif level == AlertLevel.CAUTION:
            logger.warning("CAUTION", **log_data)
        elif level == AlertLevel.WARNING:
            logger.warning("WARNING", **log_data)
        else:
            logger.info("INFO", **log_data)

    def _send_telegram(self, message: str):
        """Send message to Telegram."""
        if self.telegram_notifier:
            try:
                self.telegram_notifier.send(message)
            except Exception as e:
                logger.error("Telegram send failed", error=str(e))
        else:
            logger.warning("Telegram notifier not configured", message=message)

    def send_daily_summary(self):
        """Send daily summary to Telegram."""
        cfg = self.config["alerts"]["daily_summary"]
        if not cfg["enabled"]:
            return

        summary = f"""
📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}

💰 PnL: ${self.trading_state.daily_pnl_usd:,.2f}
📈 Positions: {len(self.trading_state.positions)}
🤖 Active Models: {self.trading_state.active_models}
⚠️ Circuit Breakers: {'ACTIVE' if self.circuit_state.is_paused else 'OK'}
"""
        self._send_telegram(summary)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_model(self, metrics: Dict[str, float]) -> tuple[bool, str]:
        """
        Validate a model against configured thresholds.

        Returns:
            Tuple of (passed, reason).
        """
        cfg = self.config["validation"]["filter_1_basic"]

        checks = [
            (metrics.get("sharpe_ratio", 0) >= cfg["min_sharpe_ratio"],
             f"Sharpe {metrics.get('sharpe_ratio', 0):.2f} < {cfg['min_sharpe_ratio']}"),

            (metrics.get("max_drawdown_pct", 100) <= cfg["max_drawdown_pct"],
             f"MaxDD {metrics.get('max_drawdown_pct', 100):.1f}% > {cfg['max_drawdown_pct']}%"),

            (metrics.get("win_rate_pct", 0) >= cfg["min_win_rate_pct"],
             f"WinRate {metrics.get('win_rate_pct', 0):.1f}% < {cfg['min_win_rate_pct']}%"),

            (metrics.get("profit_factor", 0) >= cfg["min_profit_factor"],
             f"PF {metrics.get('profit_factor', 0):.2f} < {cfg['min_profit_factor']}"),

            (metrics.get("trade_count", 0) >= cfg["min_trades"],
             f"Trades {metrics.get('trade_count', 0)} < {cfg['min_trades']}"),
        ]

        for passed, reason in checks:
            if not passed:
                return False, reason

        return True, "All checks passed"

    # =========================================================================
    # STATE UPDATES
    # =========================================================================

    def update_drawdown(self, current_equity: float, peak_equity: float):
        """Update drawdown state."""
        if peak_equity > 0:
            self.circuit_state.drawdown_pct = (peak_equity - current_equity) / peak_equity * 100

    def update_daily_pnl(self, pnl: float):
        """Update daily PnL."""
        self.trading_state.daily_pnl_usd = pnl

    def record_trade_result(self, won: bool):
        """Record trade result for consecutive losses tracking."""
        if won:
            self.circuit_state.consecutive_losses = 0
        else:
            self.circuit_state.consecutive_losses += 1

    def update_data_timestamp(self):
        """Mark data as fresh."""
        self.circuit_state.last_data_timestamp = datetime.now()

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.trading_state.daily_pnl_usd = 0.0
        self.circuit_state.is_paused = False
        self.circuit_state.pause_reason = None
        self.circuit_state.exposure_reduction_pct = 0.0
        logger.info("Daily reset completed")


# Singleton instance
_controller: Optional[AutonomousController] = None


def get_controller() -> AutonomousController:
    """Get or create the autonomous controller singleton."""
    global _controller
    if _controller is None:
        _controller = AutonomousController()
    return _controller
