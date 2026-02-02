"""
Data Pipeline Package

Production-ready multi-source data ingestion system.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     INGESTION LAYER                         │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
    │  │ Databento│  │ Binance  │  │ Polygon  │  │   IBKR   │    │
    │  │ (stocks) │  │ (crypto) │  │ (backup) │  │  (live)  │    │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
    └───────┼─────────────┼─────────────┼─────────────┼──────────┘
            │             │             │             │
            ▼             ▼             ▼             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    QUALITY CONTROL                          │
    │  • Timestamp validation (market hours)                      │
    │  • Outlier detection (> 3σ)                                 │
    │  • Volume validation (zero = suspicious)                    │
    │  • Confidence scoring per source                            │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    RECONCILIATION                           │
    │  • Conflict resolution (use higher confidence)              │
    │  • Gap filling from secondary sources                       │
    │  • Audit logging (all discrepancies)                        │
    │  • Alerts (discrepancy > 0.1%)                              │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      STORAGE                                │
    │  ┌──────────────────┐  ┌──────────────────┐                │
    │  │   TimescaleDB    │  │     Parquet      │                │
    │  │  • raw (30 days) │  │  • backtest opt  │                │
    │  │  • clean (∞)     │  │  • NautilusCatalog│                │
    │  └──────────────────┘  └──────────────────┘                │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     MONITORING                              │
    │  • Source health checks                                     │
    │  • Gap detection dashboard                                  │
    │  • Telegram alerts (source down > 5 min)                    │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from data.pipeline import DataPipeline, PipelineConfig

    config = PipelineConfig.from_env()
    pipeline = DataPipeline(config)

    # Process single symbol
    result = pipeline.process("BTCUSDT", timeframe="1h", start_date="2020-01-01")

    # Batch processing
    results = pipeline.run_batch(
        symbols=["SPY", "QQQ", "BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h", "1d"],
    )

    # Get monitoring dashboard
    status = pipeline.get_status()
"""

from data.pipeline.config import PipelineConfig, SourceConfig, AssetType
from data.pipeline.manager import DataPipeline

__all__ = [
    "PipelineConfig",
    "SourceConfig",
    "AssetType",
    "DataPipeline",
]
