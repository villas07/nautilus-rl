"""
R-010: Kelly Criterion Bet Sizing.

Implements optimal position sizing based on the Kelly Criterion.
Uses a fractional Kelly approach for risk management.

Reference:
- López de Prado - Advances in Financial Machine Learning, Ch. 10
- Ed Thorp - The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
from enum import Enum


class KellyMethod(Enum):
    """Kelly calculation methods."""
    SIMPLE = "simple"           # Basic Kelly: f = (bp - q) / b
    CONTINUOUS = "continuous"   # For continuous returns: f = (μ - r) / σ²
    SHARPE = "sharpe"          # Kelly from Sharpe: f = μ / σ²


@dataclass
class KellyConfig:
    """Configuration for Kelly Criterion sizing."""

    # Kelly fraction (conservative = 0.25, aggressive = 1.0)
    kelly_fraction: float = 0.25  # Default: 1/4 Kelly

    # Minimum probability to trade
    min_probability: float = 0.52

    # Method for calculation
    method: KellyMethod = KellyMethod.SHARPE

    # Position limits
    max_position_pct: float = 0.20  # Max 20% of capital per trade
    min_position_pct: float = 0.01  # Min 1% of capital

    # Lookback for parameter estimation
    lookback_trades: int = 100

    # Risk-free rate for excess return calculation
    risk_free_rate: float = 0.04  # 4% annual

    # Decay for recent trades (more weight on recent)
    use_time_decay: bool = True
    decay_halflife: int = 20  # Trades

    # Volatility adjustment
    vol_target: Optional[float] = None  # Target volatility (e.g., 0.15 = 15%)

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0.5 < self.min_probability < 1:
            raise ValueError("min_probability must be in (0.5, 1)")


@dataclass
class KellyResult:
    """Result of Kelly sizing calculation."""

    # Optimal fraction of capital to risk
    kelly_fraction: float

    # Adjusted fraction (after applying conservative multiplier)
    adjusted_fraction: float

    # Estimated win probability
    win_probability: float

    # Average win/loss ratio
    win_loss_ratio: float

    # Expected edge (positive = favorable)
    edge: float

    # Confidence in estimate (0-1)
    confidence: float

    # Whether to trade
    should_trade: bool

    # Recommended position size as fraction of capital
    position_size: float

    # Method used
    method: KellyMethod = KellyMethod.SHARPE

    # Additional stats
    stats: Dict = field(default_factory=dict)


class KellyCriterion:
    """
    Kelly Criterion position sizing.

    Calculates optimal bet size based on:
    - Historical win rate
    - Average win/loss ratio
    - Confidence in probability estimate
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        """Initialize Kelly calculator."""
        self.config = config or KellyConfig()

        # Trade history for estimation
        self._trade_history: List[Dict] = []

    def add_trade(
        self,
        pnl: float,
        entry_price: float,
        exit_price: float,
        direction: int,  # 1=long, -1=short
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add a completed trade to history.

        Args:
            pnl: Profit/loss in dollars
            entry_price: Entry price
            exit_price: Exit price
            direction: 1 for long, -1 for short
            timestamp: Trade timestamp (optional)
        """
        returns = (exit_price - entry_price) / entry_price * direction

        self._trade_history.append({
            "pnl": pnl,
            "returns": returns,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "timestamp": timestamp or len(self._trade_history),
            "win": pnl > 0,
        })

        # Keep only recent trades
        max_history = self.config.lookback_trades * 2
        if len(self._trade_history) > max_history:
            self._trade_history = self._trade_history[-max_history:]

    def add_trades_from_returns(self, returns: np.ndarray) -> None:
        """
        Add trades from array of returns.

        Args:
            returns: Array of trade returns (as fractions, e.g., 0.02 = 2%)
        """
        for i, ret in enumerate(returns):
            self._trade_history.append({
                "pnl": ret,  # Use return as pnl proxy
                "returns": ret,
                "direction": 1 if ret >= 0 else -1,
                "entry_price": 100,
                "exit_price": 100 * (1 + ret),
                "timestamp": i,
                "win": ret > 0,
            })

    def calculate(
        self,
        probability: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> KellyResult:
        """
        Calculate Kelly optimal position size.

        Args:
            probability: Estimated win probability (if None, uses history)
            win_loss_ratio: Average win / average loss (if None, uses history)

        Returns:
            KellyResult with sizing recommendation
        """
        # Get parameters from history or use provided
        if probability is None or win_loss_ratio is None:
            est_prob, est_ratio, confidence = self._estimate_parameters()
            probability = probability or est_prob
            win_loss_ratio = win_loss_ratio or est_ratio
        else:
            confidence = 0.8  # Default confidence when parameters provided

        # Calculate Kelly fraction based on method
        # When parameters are explicitly provided, always use SIMPLE method
        # History-based methods only work when we have trade history
        if probability is not None and win_loss_ratio is not None:
            # Parameters provided - use simple formula
            kelly_f = self._kelly_simple(probability, win_loss_ratio)
        elif self.config.method == KellyMethod.SIMPLE:
            kelly_f = self._kelly_simple(probability, win_loss_ratio)
        elif self.config.method == KellyMethod.CONTINUOUS:
            kelly_f = self._kelly_continuous()
        else:  # SHARPE
            kelly_f = self._kelly_sharpe()

        # Calculate edge
        edge = probability * win_loss_ratio - (1 - probability)

        # Apply fractional Kelly
        adjusted_f = kelly_f * self.config.kelly_fraction

        # Apply confidence adjustment
        adjusted_f *= confidence

        # Apply position limits
        position_size = np.clip(
            adjusted_f,
            self.config.min_position_pct,
            self.config.max_position_pct
        )

        # Check if should trade
        should_trade = (
            probability >= self.config.min_probability
            and edge > 0
            and kelly_f > 0
        )

        if not should_trade:
            position_size = 0.0

        return KellyResult(
            kelly_fraction=kelly_f,
            adjusted_fraction=adjusted_f,
            win_probability=probability,
            win_loss_ratio=win_loss_ratio,
            edge=edge,
            confidence=confidence,
            should_trade=should_trade,
            position_size=position_size,
            method=self.config.method,
            stats={
                "n_trades": len(self._trade_history),
                "recent_trades": min(self.config.lookback_trades, len(self._trade_history)),
            }
        )

    def calculate_from_predictions(
        self,
        predicted_prob: float,
        predicted_return: float,
        predicted_vol: Optional[float] = None,
    ) -> KellyResult:
        """
        Calculate Kelly size from model predictions.

        Args:
            predicted_prob: Probability of winning trade
            predicted_return: Expected return if correct
            predicted_vol: Expected volatility (optional)

        Returns:
            KellyResult with sizing recommendation
        """
        # Estimate loss based on return and symmetry assumption
        # Could be improved with actual loss predictions
        estimated_loss = predicted_return  # Assume symmetric for simplicity

        win_loss_ratio = predicted_return / estimated_loss if estimated_loss > 0 else 1.0

        result = self.calculate(
            probability=predicted_prob,
            win_loss_ratio=win_loss_ratio,
        )

        # Apply volatility targeting if configured
        if predicted_vol is not None and self.config.vol_target is not None:
            vol_scalar = self.config.vol_target / predicted_vol
            result.position_size = np.clip(
                result.position_size * vol_scalar,
                self.config.min_position_pct,
                self.config.max_position_pct
            )

        return result

    def _kelly_simple(self, probability: float, win_loss_ratio: float) -> float:
        """
        Simple Kelly formula: f* = (bp - q) / b

        Where:
        - b = win/loss ratio (odds)
        - p = probability of winning
        - q = probability of losing (1 - p)
        """
        b = win_loss_ratio
        p = probability
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly_f = (b * p - q) / b
        return max(0.0, kelly_f)

    def _kelly_continuous(self) -> float:
        """
        Kelly for continuous returns: f* = (μ - r) / σ²

        Where:
        - μ = expected return
        - r = risk-free rate
        - σ² = variance of returns
        """
        if len(self._trade_history) < 10:
            return 0.0

        returns = self._get_weighted_returns()

        mu = np.mean(returns)
        sigma_sq = np.var(returns)
        r = self.config.risk_free_rate / 252  # Daily risk-free

        if sigma_sq <= 0:
            return 0.0

        kelly_f = (mu - r) / sigma_sq
        return max(0.0, min(1.0, kelly_f))

    def _kelly_sharpe(self) -> float:
        """
        Kelly from Sharpe ratio: f* = μ / σ²

        Simplified version assuming risk-free = 0 or included in μ.
        """
        if len(self._trade_history) < 10:
            return 0.0

        returns = self._get_weighted_returns()

        mu = np.mean(returns)
        sigma_sq = np.var(returns)

        if sigma_sq <= 0:
            return 0.0

        kelly_f = mu / sigma_sq
        return max(0.0, min(1.0, kelly_f))

    def _estimate_parameters(self) -> Tuple[float, float, float]:
        """
        Estimate win probability and win/loss ratio from history.

        Returns:
            (probability, win_loss_ratio, confidence)
        """
        if len(self._trade_history) < 5:
            # Not enough data - return conservative defaults
            return 0.5, 1.0, 0.0

        # Get recent trades
        recent = self._trade_history[-self.config.lookback_trades:]

        # Apply time decay if enabled
        if self.config.use_time_decay:
            weights = self._get_decay_weights(len(recent))
        else:
            weights = np.ones(len(recent))

        # Calculate weighted win probability
        wins = np.array([1 if t["win"] else 0 for t in recent])
        win_prob = np.average(wins, weights=weights)

        # Calculate win/loss ratio
        win_returns = [t["returns"] for t in recent if t["win"]]
        loss_returns = [-t["returns"] for t in recent if not t["win"]]

        avg_win = np.mean(win_returns) if win_returns else 0.01
        avg_loss = np.mean(loss_returns) if loss_returns else 0.01

        # Avoid division by zero
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Confidence based on sample size
        n_trades = len(recent)
        confidence = min(1.0, n_trades / self.config.lookback_trades)

        return win_prob, win_loss_ratio, confidence

    def _get_weighted_returns(self) -> np.ndarray:
        """Get returns with optional time decay weighting."""
        recent = self._trade_history[-self.config.lookback_trades:]
        returns = np.array([t["returns"] for t in recent])

        if not self.config.use_time_decay:
            return returns

        # Apply decay weights
        weights = self._get_decay_weights(len(returns))

        # Weight returns (approximate by repeating)
        weighted_returns = returns * np.sqrt(weights)  # sqrt to not double-count in variance

        return weighted_returns

    def _get_decay_weights(self, n: int) -> np.ndarray:
        """Get exponential decay weights (more recent = higher weight)."""
        decay = np.exp(-np.arange(n)[::-1] * np.log(2) / self.config.decay_halflife)
        return decay / decay.sum() * n  # Normalize to sum to n

    def get_stats(self) -> Dict:
        """Get current statistics from trade history."""
        if len(self._trade_history) < 5:
            return {"status": "insufficient_data", "n_trades": len(self._trade_history)}

        recent = self._trade_history[-self.config.lookback_trades:]
        returns = np.array([t["returns"] for t in recent])

        wins = sum(1 for t in recent if t["win"])

        return {
            "n_trades": len(self._trade_history),
            "recent_trades": len(recent),
            "win_rate": wins / len(recent),
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "sharpe": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            "max_return": np.max(returns),
            "min_return": np.min(returns),
        }

    def reset(self) -> None:
        """Clear trade history."""
        self._trade_history = []


def calculate_kelly_size(
    win_probability: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Convenience function for simple Kelly calculation.

    Args:
        win_probability: Probability of winning (0.5-1.0)
        win_loss_ratio: Average win / average loss
        kelly_fraction: Fraction of full Kelly to use (0-1)

    Returns:
        Optimal position size as fraction of capital
    """
    if win_probability <= 0.5:
        return 0.0

    b = win_loss_ratio
    p = win_probability
    q = 1 - p

    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return 0.0

    return full_kelly * kelly_fraction


def optimal_f_from_trades(returns: np.ndarray) -> float:
    """
    Calculate optimal f from trade returns.

    This is the empirical method: find f that maximizes geometric mean.

    Args:
        returns: Array of trade returns (as fractions)

    Returns:
        Optimal fraction (0-1)
    """
    if len(returns) < 10:
        return 0.0

    # Try different f values
    f_values = np.linspace(0.01, 0.5, 50)

    best_f = 0.0
    best_growth = 0.0

    for f in f_values:
        # Calculate growth factor for each trade
        growth_factors = 1 + f * returns

        # Skip if any would result in ruin
        if np.any(growth_factors <= 0):
            continue

        # Geometric mean
        geometric_mean = np.exp(np.mean(np.log(growth_factors)))

        if geometric_mean > best_growth:
            best_growth = geometric_mean
            best_f = f

    return best_f
