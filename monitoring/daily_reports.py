"""
Daily Reports Generator (R-013)

Generates and sends daily trading reports:
- Performance summary
- Trade analysis
- Risk metrics
- Model performance
- Anomalies detected

Reference: EVAL-002, governance/evaluations/EVAL-002_gaps_analysis.md
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import json

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class DailyReportConfig:
    """Configuration for daily reports."""

    report_hour: int = 18  # 6 PM
    include_trades: bool = True
    include_risk: bool = True
    include_models: bool = True
    include_anomalies: bool = True
    telegram_enabled: bool = True
    save_to_file: bool = True
    report_dir: str = "reports/"


@dataclass
class TradeStats:
    """Trade statistics for the day."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class RiskStats:
    """Risk statistics for the day."""

    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    daily_var: float = 0.0  # Value at Risk
    max_exposure: float = 0.0
    circuit_breaker_triggers: int = 0
    position_limit_hits: int = 0


@dataclass
class ModelStats:
    """Model performance statistics."""

    total_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    signal_distribution: Dict[str, int] = field(default_factory=dict)
    best_model: str = ""
    worst_model: str = ""


@dataclass
class DailyReport:
    """Complete daily report."""

    report_date: date
    timestamp: datetime

    # Summary
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    daily_return: float = 0.0
    sharpe_daily: float = 0.0

    # Detailed stats
    trade_stats: TradeStats = field(default_factory=TradeStats)
    risk_stats: RiskStats = field(default_factory=RiskStats)
    model_stats: ModelStats = field(default_factory=ModelStats)

    # Anomalies
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_date": self.report_date.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "starting_equity": self.starting_equity,
                "ending_equity": self.ending_equity,
                "daily_return": self.daily_return,
                "sharpe_daily": self.sharpe_daily,
            },
            "trade_stats": {
                "total_trades": self.trade_stats.total_trades,
                "winning_trades": self.trade_stats.winning_trades,
                "losing_trades": self.trade_stats.losing_trades,
                "total_pnl": self.trade_stats.total_pnl,
                "win_rate": self.trade_stats.win_rate,
                "profit_factor": self.trade_stats.profit_factor,
            },
            "risk_stats": {
                "max_drawdown": self.risk_stats.max_drawdown,
                "current_drawdown": self.risk_stats.current_drawdown,
                "daily_var": self.risk_stats.daily_var,
                "circuit_breaker_triggers": self.risk_stats.circuit_breaker_triggers,
            },
            "model_stats": {
                "total_predictions": self.model_stats.total_predictions,
                "accuracy": self.model_stats.accuracy,
                "avg_confidence": self.model_stats.avg_confidence,
            },
            "anomalies": self.anomalies,
            "warnings": self.warnings,
        }


class DailyReportGenerator:
    """
    Generates daily trading reports.

    Collects data from various sources and produces
    a comprehensive daily summary.
    """

    def __init__(self, config: Optional[DailyReportConfig] = None):
        """Initialize report generator."""
        self.config = config or DailyReportConfig()
        self._trades_today: List[Dict[str, Any]] = []
        self._predictions_today: List[Dict[str, Any]] = []
        self._equity_snapshots: List[float] = []
        self._starting_equity: float = 100000.0
        self._current_equity: float = 100000.0

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a trade for the daily report."""
        self._trades_today.append({
            **trade,
            "timestamp": datetime.now(),
        })

    def record_prediction(self, prediction: Dict[str, Any]) -> None:
        """Record a model prediction."""
        self._predictions_today.append({
            **prediction,
            "timestamp": datetime.now(),
        })

    def record_equity(self, equity: float) -> None:
        """Record equity snapshot."""
        self._equity_snapshots.append(equity)
        self._current_equity = equity

    def set_starting_equity(self, equity: float) -> None:
        """Set starting equity for the day."""
        self._starting_equity = equity

    def generate_report(self, report_date: Optional[date] = None) -> DailyReport:
        """Generate the daily report."""
        report_date = report_date or date.today()

        report = DailyReport(
            report_date=report_date,
            timestamp=datetime.now(),
            starting_equity=self._starting_equity,
            ending_equity=self._current_equity,
        )

        # Calculate daily return
        if self._starting_equity > 0:
            report.daily_return = (
                (self._current_equity - self._starting_equity)
                / self._starting_equity
            )

        # Trade stats
        report.trade_stats = self._calculate_trade_stats()

        # Risk stats
        report.risk_stats = self._calculate_risk_stats()

        # Model stats
        report.model_stats = self._calculate_model_stats()

        # Anomalies and warnings
        report.anomalies = self._detect_anomalies(report)
        report.warnings = self._generate_warnings(report)

        return report

    def _calculate_trade_stats(self) -> TradeStats:
        """Calculate trade statistics."""
        stats = TradeStats()

        if not self._trades_today:
            return stats

        pnls = [t.get("pnl", 0) for t in self._trades_today]

        stats.total_trades = len(self._trades_today)
        stats.winning_trades = sum(1 for p in pnls if p > 0)
        stats.losing_trades = sum(1 for p in pnls if p < 0)
        stats.total_pnl = sum(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        stats.avg_win = np.mean(wins) if wins else 0.0
        stats.avg_loss = np.mean(losses) if losses else 0.0
        stats.largest_win = max(wins) if wins else 0.0
        stats.largest_loss = min(losses) if losses else 0.0

        stats.win_rate = (
            stats.winning_trades / stats.total_trades
            if stats.total_trades > 0 else 0.0
        )

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        stats.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return stats

    def _calculate_risk_stats(self) -> RiskStats:
        """Calculate risk statistics."""
        stats = RiskStats()

        if len(self._equity_snapshots) < 2:
            return stats

        equities = np.array(self._equity_snapshots)

        # Drawdown calculation
        peak = np.maximum.accumulate(equities)
        drawdowns = (peak - equities) / peak

        stats.max_drawdown = float(np.max(drawdowns))
        stats.current_drawdown = float(drawdowns[-1])

        # Daily VaR (95% confidence)
        returns = np.diff(equities) / equities[:-1]
        if len(returns) > 1:
            stats.daily_var = float(np.percentile(returns, 5))

        return stats

    def _calculate_model_stats(self) -> ModelStats:
        """Calculate model performance statistics."""
        stats = ModelStats()

        if not self._predictions_today:
            return stats

        stats.total_predictions = len(self._predictions_today)

        # Signal distribution
        signals = [p.get("signal", 0) for p in self._predictions_today]
        stats.signal_distribution = {
            "buy": sum(1 for s in signals if s > 0),
            "hold": sum(1 for s in signals if s == 0),
            "sell": sum(1 for s in signals if s < 0),
        }

        # Average confidence
        confidences = [p.get("confidence", 0) for p in self._predictions_today]
        stats.avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Accuracy (if we have outcomes)
        correct = sum(
            1 for p in self._predictions_today
            if p.get("correct", False)
        )
        stats.accuracy = correct / stats.total_predictions

        return stats

    def _detect_anomalies(self, report: DailyReport) -> List[str]:
        """Detect anomalies in the day's activity."""
        anomalies = []

        # Large daily loss
        if report.daily_return < -0.03:  # 3% loss
            anomalies.append(
                f"Large daily loss: {report.daily_return:.2%}"
            )

        # High drawdown
        if report.risk_stats.max_drawdown > 0.10:  # 10% drawdown
            anomalies.append(
                f"High drawdown: {report.risk_stats.max_drawdown:.2%}"
            )

        # Unusual trade count
        avg_trades = 10  # Expected average
        if report.trade_stats.total_trades > avg_trades * 3:
            anomalies.append(
                f"Unusual trade volume: {report.trade_stats.total_trades} trades"
            )

        # Low model confidence
        if report.model_stats.avg_confidence < 0.3:
            anomalies.append(
                f"Low model confidence: {report.model_stats.avg_confidence:.2%}"
            )

        return anomalies

    def _generate_warnings(self, report: DailyReport) -> List[str]:
        """Generate warnings based on report."""
        warnings = []

        # Win rate too low
        if (report.trade_stats.total_trades >= 5
            and report.trade_stats.win_rate < 0.40):
            warnings.append(
                f"Low win rate: {report.trade_stats.win_rate:.1%}"
            )

        # No trades
        if report.trade_stats.total_trades == 0:
            warnings.append("No trades executed today")

        # Approaching drawdown limit
        if report.risk_stats.current_drawdown > 0.07:
            warnings.append(
                f"Drawdown approaching limit: {report.risk_stats.current_drawdown:.1%}"
            )

        return warnings

    def format_telegram_report(self, report: DailyReport) -> str:
        """Format report for Telegram."""
        # Determine overall status
        if report.daily_return > 0.01:
            status = "🟢"
        elif report.daily_return < -0.01:
            status = "🔴"
        else:
            status = "🟡"

        lines = [
            f"{status} *Daily Report - {report.report_date}*",
            "",
            "*📊 Performance*",
            f"  Daily Return: {report.daily_return:+.2%}",
            f"  Ending Equity: ${report.ending_equity:,.2f}",
            f"  PnL: ${report.trade_stats.total_pnl:+,.2f}",
            "",
            "*📈 Trades*",
            f"  Total: {report.trade_stats.total_trades}",
            f"  Win Rate: {report.trade_stats.win_rate:.1%}",
            f"  Profit Factor: {report.trade_stats.profit_factor:.2f}",
            "",
            "*⚠️ Risk*",
            f"  Max Drawdown: {report.risk_stats.max_drawdown:.2%}",
            f"  Current DD: {report.risk_stats.current_drawdown:.2%}",
        ]

        if report.anomalies:
            lines.extend([
                "",
                "*🚨 Anomalies*",
            ])
            for anomaly in report.anomalies:
                lines.append(f"  • {anomaly}")

        if report.warnings:
            lines.extend([
                "",
                "*⚡ Warnings*",
            ])
            for warning in report.warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)

    def save_report(self, report: DailyReport) -> str:
        """Save report to file."""
        import os

        os.makedirs(self.config.report_dir, exist_ok=True)

        filename = f"{self.config.report_dir}report_{report.report_date}.json"

        with open(filename, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Daily report saved: {filename}")
        return filename

    async def send_telegram_report(self, report: DailyReport) -> None:
        """Send report via Telegram."""
        if not self.config.telegram_enabled:
            return

        try:
            from live.telegram_alerts import get_alerts, AlertLevel

            alerts = get_alerts()
            message = self.format_telegram_report(report)
            await alerts.send_message(message, AlertLevel.INFO)

            logger.info("Daily report sent to Telegram")

        except Exception as e:
            logger.error(f"Failed to send Telegram report: {e}")

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._trades_today = []
        self._predictions_today = []
        self._equity_snapshots = []
        self._starting_equity = self._current_equity

        logger.info("Daily report counters reset")


# Singleton instance
_report_generator: Optional[DailyReportGenerator] = None


def get_report_generator() -> DailyReportGenerator:
    """Get or create the report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = DailyReportGenerator()
    return _report_generator


async def generate_and_send_daily_report() -> DailyReport:
    """Convenience function to generate and send daily report."""
    generator = get_report_generator()
    report = generator.generate_report()

    # Save to file
    generator.save_report(report)

    # Send to Telegram
    await generator.send_telegram_report(report)

    # Reset for next day
    generator.reset_daily()

    return report
