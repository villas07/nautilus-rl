"""
Filter 5: Paper Trading Validation

Final validation through live paper trading:
- 2-4 weeks of paper trading
- Real-time data and execution
- Validates slippage and execution quality
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import json

import structlog

logger = structlog.get_logger()


@dataclass
class PaperTradingCriteria:
    """Criteria for paper trading validation."""

    min_trading_days: int = 10  # Minimum days of paper trading
    max_trading_days: int = 30  # Maximum days before final decision

    min_sharpe_paper: float = 0.5  # Lower bar for short period
    max_drawdown_paper: float = 0.10  # Stricter for live trading
    min_trades_paper: int = 10  # Minimum trades executed
    max_slippage_pct: float = 0.5  # Max average slippage


@dataclass
class PaperTradeRecord:
    """Record of a paper trade."""

    timestamp: str
    agent_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    expected_price: float
    executed_price: float
    slippage: float
    status: str  # filled, cancelled, rejected


class PaperTradingFilter:
    """
    Filter 5: Paper trading validation.

    Runs agents in paper trading mode to validate real-time performance.
    """

    def __init__(
        self,
        criteria: Optional[PaperTradingCriteria] = None,
        models_dir: str = "/app/models",
        paper_log_dir: str = "/app/logs/paper_trading",
    ):
        """
        Initialize filter.

        Args:
            criteria: Validation criteria.
            models_dir: Directory containing trained models.
            paper_log_dir: Directory for paper trading logs.
        """
        self.criteria = criteria or PaperTradingCriteria()
        self.models_dir = Path(models_dir)
        self.paper_log_dir = Path(paper_log_dir)
        self.paper_log_dir.mkdir(parents=True, exist_ok=True)

    def start_paper_trading(
        self,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Start paper trading session for agents.

        This would integrate with the live trading system in paper mode.
        Returns session info.
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.paper_log_dir / session_id
        session_dir.mkdir(exist_ok=True)

        session_info = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "agents": agent_ids,
            "status": "running",
            "trades": [],
        }

        # Save session info
        with open(session_dir / "session.json", "w") as f:
            json.dump(session_info, f, indent=2)

        logger.info(
            f"Started paper trading session {session_id} with {len(agent_ids)} agents"
        )

        return session_info

    def get_paper_trading_results(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get results from a paper trading session.

        Args:
            session_id: Session identifier.

        Returns:
            Session results or None if not found.
        """
        session_dir = self.paper_log_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            return json.load(f)

    def validate_from_session(
        self,
        session_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate agents from a completed paper trading session.

        Args:
            session_id: Session identifier.

        Returns:
            Dict mapping agent_id to validation results.
        """
        session = self.get_paper_trading_results(session_id)
        if session is None:
            return {}

        trades = session.get("trades", [])
        if not trades:
            logger.warning(f"No trades found in session {session_id}")
            return {}

        # Group trades by agent
        agent_trades: Dict[str, List[Dict]] = {}
        for trade in trades:
            agent_id = trade.get("agent_id")
            if agent_id:
                if agent_id not in agent_trades:
                    agent_trades[agent_id] = []
                agent_trades[agent_id].append(trade)

        # Validate each agent
        results = {}
        for agent_id, trades in agent_trades.items():
            results[agent_id] = self._validate_agent_trades(agent_id, trades)

        return results

    def validate(
        self,
        agent_id: str,
        trades: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a single agent from paper trading.

        Args:
            agent_id: Agent identifier.
            trades: List of trade records.
            session_id: Or session ID to load trades from.

        Returns:
            Validation results.
        """
        if trades is None and session_id:
            session = self.get_paper_trading_results(session_id)
            if session:
                all_trades = session.get("trades", [])
                trades = [t for t in all_trades if t.get("agent_id") == agent_id]

        if not trades:
            return {
                "agent_id": agent_id,
                "passed": False,
                "reason": "No trades found",
            }

        return self._validate_agent_trades(agent_id, trades)

    def _validate_agent_trades(
        self,
        agent_id: str,
        trades: List[Dict],
    ) -> Dict[str, Any]:
        """Validate agent based on trade history."""
        if len(trades) < self.criteria.min_trades_paper:
            return {
                "agent_id": agent_id,
                "passed": False,
                "reason": f"Too few trades: {len(trades)} < {self.criteria.min_trades_paper}",
                "num_trades": len(trades),
            }

        # Calculate metrics from trades
        pnls = []
        slippages = []

        for trade in trades:
            if "pnl" in trade:
                pnls.append(trade["pnl"])
            if "slippage" in trade:
                slippages.append(abs(trade["slippage"]))

        # Calculate Sharpe-like metric (simplified)
        if pnls:
            import numpy as np
            pnls_arr = np.array(pnls)
            if pnls_arr.std() > 0:
                sharpe = np.sqrt(252) * pnls_arr.mean() / pnls_arr.std()
            else:
                sharpe = 0.0

            # Calculate drawdown
            cumulative = np.cumsum(pnls_arr)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / (running_max + 1e-10)
            max_dd = drawdowns.max()

            total_return = sum(pnls)
        else:
            sharpe = 0.0
            max_dd = 0.0
            total_return = 0.0

        # Average slippage
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

        # Check criteria
        failures = []

        if sharpe < self.criteria.min_sharpe_paper:
            failures.append(
                f"Sharpe {sharpe:.2f} < {self.criteria.min_sharpe_paper}"
            )

        if max_dd > self.criteria.max_drawdown_paper:
            failures.append(
                f"Drawdown {max_dd:.1%} > {self.criteria.max_drawdown_paper:.1%}"
            )

        if avg_slippage > self.criteria.max_slippage_pct:
            failures.append(
                f"Slippage {avg_slippage:.2%} > {self.criteria.max_slippage_pct:.1%}"
            )

        passed = len(failures) == 0

        return {
            "agent_id": agent_id,
            "passed": passed,
            "failures": failures,
            "metrics": {
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_return": total_return,
                "avg_slippage": avg_slippage,
                "num_trades": len(trades),
            },
        }

    def record_trade(
        self,
        session_id: str,
        trade: Dict[str, Any],
    ) -> None:
        """
        Record a trade to the session log.

        Called by the live trading system during paper trading.
        """
        session_dir = self.paper_log_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            logger.warning(f"Session {session_id} not found")
            return

        with open(session_file) as f:
            session = json.load(f)

        trade["recorded_at"] = datetime.now().isoformat()
        session["trades"].append(trade)

        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)

    def get_active_sessions(self) -> List[str]:
        """Get list of active paper trading sessions."""
        sessions = []

        for session_dir in self.paper_log_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file) as f:
                            session = json.load(f)
                        if session.get("status") == "running":
                            sessions.append(session_dir.name)
                    except Exception:
                        pass

        return sessions


def filter_paper_trading(
    agent_ids: List[str],
    session_id: str,
    models_dir: str = "/app/models",
) -> List[str]:
    """
    Convenience function to filter agents from paper trading session.

    Returns list of agent IDs that pass.
    """
    filter_obj = PaperTradingFilter(models_dir=models_dir)
    results = filter_obj.validate_from_session(session_id)

    return [
        agent_id for agent_id, result in results.items()
        if result.get("passed", False)
    ]
