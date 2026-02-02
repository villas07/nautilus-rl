"""Observation space builder for the trading environment."""

from typing import Optional, List
import numpy as np
import pandas as pd


class ObservationBuilder:
    """
    Builds observation vectors from market data.

    Features (45 total):
        Price Features (15):
            - Last 5 returns
            - Mean return
            - Return volatility
            - MA crossovers (5/10, 10/20)
            - Price vs MAs
            - Bollinger Band position
            - High/Low range position

        Volume Features (5):
            - Volume ratio (current/average)
            - Volume trend
            - Volume volatility
            - VWAP distance
            - OBV direction

        Technical Indicators (10):
            - RSI (normalized)
            - MACD signal
            - ADX
            - CCI
            - Stochastic K/D
            - ATR ratio
            - Momentum 5/10

        Position Features (5):
            - Current position
            - Is long flag
            - Is short flag
            - PnL on position
            - Time in position

        Time Features (4):
            - Hour sin/cos
            - Day of week sin/cos

        Portfolio Features (6):
            - Portfolio return
            - Drawdown
            - Win rate
            - Profit factor
            - Trades count normalized
            - Cash ratio
    """

    def __init__(
        self,
        lookback_period: int = 20,
        num_features: int = 45,
        include_position: bool = True,
        include_time: bool = True,
    ):
        """
        Initialize the observation builder.

        Args:
            lookback_period: Number of bars to use for features.
            num_features: Total number of features in observation.
            include_position: Include position-related features.
            include_time: Include time-based features.
        """
        self.lookback_period = lookback_period
        self.num_features = num_features
        self.include_position = include_position
        self.include_time = include_time

    def build(
        self,
        data: pd.DataFrame,
        position: float = 0.0,
        portfolio_value: float = 100000.0,
        initial_capital: float = 100000.0,
    ) -> np.ndarray:
        """
        Build observation vector from market data.

        Args:
            data: DataFrame with OHLCV data (at least lookback_period rows).
            position: Current position (-1 to 1).
            portfolio_value: Current portfolio value.
            initial_capital: Starting capital.

        Returns:
            Observation vector of shape (num_features,).
        """
        features = []

        # Extract price data
        closes = data["close"].values.astype(float)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        volumes = data["volume"].values.astype(float)

        # Price features
        features.extend(self._price_features(closes, highs, lows))

        # Volume features
        features.extend(self._volume_features(closes, volumes))

        # Technical indicators
        features.extend(self._technical_features(closes, highs, lows))

        # Position features
        if self.include_position:
            features.extend(self._position_features(
                position, portfolio_value, initial_capital
            ))

        # Time features
        if self.include_time:
            features.extend(self._time_features(data))

        # Portfolio features
        features.extend(self._portfolio_features(
            portfolio_value, initial_capital
        ))

        # Pad or truncate to exact size
        features = np.array(features, dtype=np.float32)
        if len(features) < self.num_features:
            features = np.pad(features, (0, self.num_features - len(features)))
        elif len(features) > self.num_features:
            features = features[:self.num_features]

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip extreme values
        features = np.clip(features, -10.0, 10.0)

        return features

    def _price_features(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[float]:
        """Calculate price-based features."""
        features = []

        # Returns
        returns = np.diff(closes) / closes[:-1]

        # Last 5 returns
        last_returns = returns[-5:] if len(returns) >= 5 else np.zeros(5)
        features.extend(last_returns.tolist())

        # Return statistics
        features.append(np.mean(returns))
        features.append(np.std(returns))

        # Moving averages
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]

        current = closes[-1]

        # MA crossovers (normalized)
        features.append((ma5 - ma10) / ma10 if ma10 != 0 else 0)
        features.append((ma10 - ma20) / ma20 if ma20 != 0 else 0)

        # Price vs MAs
        features.append((current - ma5) / ma5 if ma5 != 0 else 0)
        features.append((current - ma20) / ma20 if ma20 != 0 else 0)

        # Bollinger Bands
        std20 = np.std(closes[-20:]) if len(closes) >= 20 else 0.01
        upper_bb = ma20 + 2 * std20
        lower_bb = ma20 - 2 * std20
        bb_width = upper_bb - lower_bb if upper_bb > lower_bb else 1.0
        features.append((current - lower_bb) / bb_width - 0.5)

        # High/Low range position
        high_20 = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
        low_20 = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
        range_20 = high_20 - low_20 if high_20 > low_20 else 1.0
        features.append((current - low_20) / range_20 - 0.5)

        return features

    def _volume_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> List[float]:
        """Calculate volume-based features."""
        features = []

        avg_volume = np.mean(volumes) if len(volumes) > 0 else 1.0

        # Volume ratio
        features.append(volumes[-1] / avg_volume if avg_volume > 0 else 1.0)

        # Volume trend (recent vs earlier)
        if len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            earlier_vol = np.mean(volumes[-10:-5])
            features.append((recent_vol - earlier_vol) / earlier_vol if earlier_vol > 0 else 0)
        else:
            features.append(0.0)

        # Volume volatility
        features.append(np.std(volumes) / avg_volume if avg_volume > 0 else 0)

        # Simple VWAP proxy
        if len(volumes) > 0 and np.sum(volumes) > 0:
            vwap = np.sum(closes * volumes) / np.sum(volumes)
            features.append((closes[-1] - vwap) / vwap if vwap > 0 else 0)
        else:
            features.append(0.0)

        # OBV direction (simplified)
        if len(closes) >= 2:
            obv_change = np.sign(closes[-1] - closes[-2]) * volumes[-1]
            features.append(np.sign(obv_change))
        else:
            features.append(0.0)

        return features

    def _technical_features(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[float]:
        """Calculate technical indicator features."""
        features = []

        # RSI
        rsi = self._calculate_rsi(closes, 14)
        features.append((rsi - 50) / 50)

        # MACD signal (simplified)
        if len(closes) >= 26:
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd = ema12 - ema26
            signal = self._ema(np.array([macd]), 9)
            features.append(np.sign(macd - signal))
        else:
            features.append(0.0)

        # ADX (simplified)
        adx = self._calculate_adx(highs, lows, closes, 14)
        features.append(adx / 50 - 1)  # Normalize around 0

        # CCI
        cci = self._calculate_cci(highs, lows, closes, 20)
        features.append(cci / 200)  # Normalize

        # Stochastic
        k, d = self._calculate_stochastic(highs, lows, closes, 14, 3)
        features.append((k - 50) / 50)
        features.append((d - 50) / 50)

        # ATR ratio
        atr = self._calculate_atr(highs, lows, closes, 14)
        features.append(atr / closes[-1] if closes[-1] > 0 else 0)

        # Momentum
        if len(closes) >= 5:
            features.append((closes[-1] - closes[-5]) / closes[-5])
        else:
            features.append(0.0)

        if len(closes) >= 10:
            features.append((closes[-1] - closes[-10]) / closes[-10])
        else:
            features.append(0.0)

        return features

    def _position_features(
        self,
        position: float,
        portfolio_value: float,
        initial_capital: float,
    ) -> List[float]:
        """Calculate position-related features."""
        features = []

        # Current position
        features.append(position)

        # Is long/short flags
        features.append(1.0 if position > 0 else 0.0)
        features.append(1.0 if position < 0 else 0.0)

        # PnL on position (simplified as portfolio return)
        pnl = (portfolio_value - initial_capital) / initial_capital
        features.append(pnl)

        # Position size normalized
        features.append(abs(position))

        return features

    def _time_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate time-based features."""
        features = []

        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"].iloc[-1])
            hour = ts.hour
            day_of_week = ts.dayofweek

            # Cyclical encoding
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(np.sin(2 * np.pi * day_of_week / 7))
            features.append(np.cos(2 * np.pi * day_of_week / 7))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    def _portfolio_features(
        self,
        portfolio_value: float,
        initial_capital: float,
    ) -> List[float]:
        """Calculate portfolio-related features."""
        features = []

        # Portfolio return
        features.append((portfolio_value - initial_capital) / initial_capital)

        # Simplified drawdown (would need peak tracking in real impl)
        features.append(0.0)

        # Placeholders for win rate, profit factor, trades count
        features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    # Helper methods for technical indicators

    def _ema(self, values: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(values) < period:
            return values[-1] if len(values) > 0 else 0.0

        multiplier = 2 / (period + 1)
        ema = values[0]
        for val in values[1:]:
            ema = (val - ema) * multiplier + ema
        return ema

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate ADX (simplified)."""
        if len(closes) < period + 1:
            return 25.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        if atr == 0:
            return 25.0

        # Simplified DX
        return 25.0 + 25 * np.random.random()  # Placeholder

    def _calculate_cci(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 20,
    ) -> float:
        """Calculate CCI."""
        if len(closes) < period:
            return 0.0

        tp = (highs + lows + closes) / 3
        tp_sma = np.mean(tp[-period:])
        mean_dev = np.mean(np.abs(tp[-period:] - tp_sma))

        if mean_dev == 0:
            return 0.0

        return (tp[-1] - tp_sma) / (0.015 * mean_dev)

    def _calculate_stochastic(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple:
        """Calculate Stochastic K and D."""
        if len(closes) < k_period:
            return 50.0, 50.0

        highest_high = np.max(highs[-k_period:])
        lowest_low = np.min(lows[-k_period:])

        if highest_high == lowest_low:
            return 50.0, 50.0

        k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)
        d = k  # Simplified

        return k, d

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate ATR."""
        if len(closes) < 2:
            return 0.0

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
