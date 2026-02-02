"""
Data Module for NautilusTrader Integration

This module provides:
- Multi-source data pipeline (pipeline/)
- Data adapters for various providers (adapters/)
- Data synchronization tools

Components:
    pipeline/: Production data pipeline with quality control
    adapters/: Source-specific adapters (Binance, Yahoo, etc.)

Quick Start:
    from data.pipeline import DataPipeline, PipelineConfig

    pipeline = DataPipeline()
    result = pipeline.process("BTCUSDT", "1h", "2020-01-01")

Legacy Adapters:
    from data.adapters.timescale_adapter import TimescaleAdapter
    from data.adapters.bar_converter import BarConverter
"""

# Pipeline exports
from data.pipeline import (
    DataPipeline,
    PipelineConfig,
    AssetType,
)

# Legacy adapter exports
from data.adapters.timescale_adapter import TimescaleAdapter
from data.adapters.bar_converter import BarConverter

__all__ = [
    # Pipeline
    "DataPipeline",
    "PipelineConfig",
    "AssetType",
    # Legacy adapters
    "TimescaleAdapter",
    "BarConverter",
]

__version__ = "1.0.0"
