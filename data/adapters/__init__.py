"""
Data Adapters

Source-specific adapters for market data:

FREE Sources (No API key required):
    - BinanceHistoricalAdapter: Crypto OHLCV (unlimited)
    - YahooFinanceAdapter: Stocks, ETFs, Crypto (unlimited)
    - CryptoCompareAdapter: Crypto (100k calls/month)
    - CoinGeckoAdapter: Crypto (10-30 calls/min)
    - FREDAdapter: Economic data (unlimited)

API Key Required:
    - AlphaVantageAdapter: Stocks, Forex (25/day free)
    - EODHistoricalDataAdapter: Europe/Asia stocks (~$80/mo)
    - DatabentoAdapter: Premium futures/stocks
    - PolygonAdapter: Stocks (5/min free tier)

Legacy:
    - TimescaleAdapter: TimescaleDB reader
    - BarConverter: Format conversion
"""

# Legacy adapters
from data.adapters.timescale_adapter import TimescaleAdapter
from data.adapters.bar_converter import BarConverter

# Free data adapters
from data.adapters.binance_adapter import BinanceHistoricalAdapter
from data.adapters.yahoo_adapter import YahooFinanceAdapter
from data.adapters.cryptocompare_adapter import CryptoCompareAdapter
from data.adapters.coingecko_adapter import CoinGeckoAdapter
from data.adapters.fred_adapter import FREDAdapter

# API key adapters
from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.eod_adapter import EODHistoricalDataAdapter

__all__ = [
    # Legacy
    "TimescaleAdapter",
    "BarConverter",
    # Free sources
    "BinanceHistoricalAdapter",
    "YahooFinanceAdapter",
    "CryptoCompareAdapter",
    "CoinGeckoAdapter",
    "FREDAdapter",
    # API key required
    "AlphaVantageAdapter",
    "EODHistoricalDataAdapter",
]
