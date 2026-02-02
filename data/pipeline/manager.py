"""
Data Pipeline Manager

Main orchestrator for the data pipeline:
1. Coordinates data fetching from multiple sources
2. Applies quality control
3. Reconciles data from different sources
4. Stores clean data in TimescaleDB and Parquet
5. Monitors pipeline health
"""

import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import structlog

from data.pipeline.config import (
    PipelineConfig,
    SourceConfig,
    AssetType,
    SourceRole,
)
from data.pipeline.base import BaseAdapter, FetchResult
from data.pipeline.quality import QualityController, QualityReport
from data.pipeline.reconciliation import DataReconciler, ReconciliationResult
from data.pipeline.storage import DataStorage
from data.pipeline.monitoring import PipelineMonitor

logger = structlog.get_logger()


@dataclass
class ProcessingResult:
    """Result of processing a single symbol."""

    symbol: str
    timeframe: str
    success: bool = False

    # Data statistics
    rows_fetched: int = 0
    rows_clean: int = 0
    rows_stored: int = 0

    # Sources
    sources_tried: List[str] = field(default_factory=list)
    sources_successful: List[str] = field(default_factory=list)
    primary_source: Optional[str] = None

    # Quality
    quality_score: float = 0.0
    quality_issues: int = 0

    # Reconciliation
    discrepancies: int = 0
    gaps_filled: int = 0

    # Timing
    duration_seconds: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "success": self.success,
            "rows_fetched": self.rows_fetched,
            "rows_clean": self.rows_clean,
            "rows_stored": self.rows_stored,
            "sources": self.sources_successful,
            "quality_score": self.quality_score,
            "discrepancies": self.discrepancies,
            "gaps_filled": self.gaps_filled,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class DataPipeline:
    """
    Main data pipeline orchestrator.

    Usage:
        pipeline = DataPipeline()

        # Process single symbol
        result = pipeline.process("BTCUSDT", "1h", "2020-01-01")

        # Batch process
        results = pipeline.run_batch(
            symbols=["SPY", "BTCUSDT"],
            timeframes=["1h", "1d"],
        )

        # Get status
        status = pipeline.get_status()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig.from_env()

        # Initialize components
        self.quality = QualityController(self.config.quality)
        self.reconciler = DataReconciler(self.config.reconciliation)
        self.storage = DataStorage(self.config.storage)
        self.monitor = PipelineMonitor(self.config.monitoring)

        # Adapter cache
        self._adapters: Dict[str, BaseAdapter] = {}

        # Set up reconciler alert callback
        self.reconciler.set_alert_callback(self._on_discrepancy_alert)

        # Validate configuration
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")

    def _get_adapter(self, source_name: str) -> Optional[BaseAdapter]:
        """Get or create adapter for a source."""
        if source_name in self._adapters:
            return self._adapters[source_name]

        source_config = self.config.sources.get(source_name)
        if not source_config or not source_config.is_available():
            return None

        try:
            # Dynamic import
            module = __import__(
                source_config.adapter_module,
                fromlist=[source_config.adapter_class],
            )
            adapter_class = getattr(module, source_config.adapter_class)
            adapter = adapter_class()
            self._adapters[source_name] = adapter
            return adapter

        except Exception as e:
            logger.error(f"Failed to load adapter {source_name}: {e}")
            return None

    def _infer_asset_type(self, symbol: str) -> AssetType:
        """Infer asset type from symbol."""
        symbol_upper = symbol.upper()

        # Crypto patterns
        if symbol_upper.endswith("USDT") or symbol_upper.endswith("BUSD"):
            return AssetType.CRYPTO
        if symbol_upper in ["BTC", "ETH", "BNB", "SOL", "XRP"]:
            return AssetType.CRYPTO
        if "-USD" in symbol_upper or "-BTC" in symbol_upper:
            return AssetType.CRYPTO

        # Forex patterns
        if "/" in symbol or symbol_upper in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            return AssetType.FOREX

        # Futures patterns
        if "=" in symbol or symbol.endswith(".FUT"):
            return AssetType.FUTURE

        # Index patterns
        if symbol.startswith("^") or symbol_upper in ["SPX", "NDX", "VIX"]:
            return AssetType.INDEX

        # ETF patterns
        etfs = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "TLT", "XLF", "XLK"}
        if symbol_upper in etfs:
            return AssetType.ETF

        # Default to stock
        return AssetType.STOCK

    def _fetch_from_source(
        self,
        source_name: str,
        source_config: SourceConfig,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str],
    ) -> FetchResult:
        """Fetch data from a single source."""
        adapter = self._get_adapter(source_name)

        if adapter is None:
            return FetchResult.empty(source_name, symbol, timeframe, "Adapter not available")

        start_time = time.time()

        try:
            result = adapter.fetch(symbol, timeframe, start_date, end_date)
            latency_ms = (time.time() - start_time) * 1000

            if result.success:
                self.monitor.record_request_success(source_name, latency_ms, symbol)
            else:
                self.monitor.record_request_failure(source_name, result.error or "Unknown", symbol)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error = str(e)
            self.monitor.record_request_failure(source_name, error, symbol)
            return FetchResult.empty(source_name, symbol, timeframe, error)

    def process(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
        save_to_db: bool = True,
        save_to_parquet: bool = True,
    ) -> ProcessingResult:
        """
        Process a single symbol through the full pipeline.

        Args:
            symbol: Symbol to process.
            timeframe: Timeframe (e.g., "1h", "1d").
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            save_to_db: Save to TimescaleDB.
            save_to_parquet: Save to Parquet.

        Returns:
            ProcessingResult with statistics.
        """
        start_time = time.time()
        result = ProcessingResult(symbol=symbol, timeframe=timeframe)

        logger.info(f"Processing {symbol} {timeframe}", start_date=start_date)

        # Determine asset type and get sources
        asset_type = self._infer_asset_type(symbol)
        sources = self.config.get_sources_for_asset(asset_type)

        if not sources:
            result.errors.append(f"No sources available for {asset_type}")
            result.duration_seconds = time.time() - start_time
            return result

        # Fetch from sources
        primary_df = pd.DataFrame()
        primary_source = None
        primary_confidence = 0.0
        secondary_data: List[Tuple[pd.DataFrame, str, float]] = []

        for source_config in sources:
            result.sources_tried.append(source_config.name)

            fetch_result = self._fetch_from_source(
                source_config.name,
                source_config,
                symbol,
                timeframe,
                start_date,
                end_date,
            )

            if fetch_result.success and not fetch_result.data.empty:
                result.sources_successful.append(source_config.name)

                if primary_df.empty:
                    # First successful source becomes primary
                    primary_df = fetch_result.data
                    primary_source = source_config.name
                    primary_confidence = source_config.confidence
                    result.primary_source = source_config.name
                    result.rows_fetched = len(primary_df)
                else:
                    # Additional sources for reconciliation
                    secondary_data.append((
                        fetch_result.data,
                        source_config.name,
                        source_config.confidence,
                    ))

                # For primary role, stop after getting data
                if source_config.role == SourceRole.PRIMARY:
                    break

            elif fetch_result.error:
                result.errors.append(f"{source_config.name}: {fetch_result.error}")

        if primary_df.empty:
            result.errors.append("Failed to fetch data from any source")
            result.duration_seconds = time.time() - start_time
            return result

        # Save raw data
        if save_to_db and primary_source:
            self.storage.save_raw(primary_df, symbol, primary_source, timeframe)

        # Quality control
        clean_df, quality_report = self.quality.validate(
            primary_df,
            symbol,
            timeframe,
            asset_type,
        )
        result.quality_score = quality_report.overall_score
        result.quality_issues = sum(quality_report.issues_by_type.values())

        # Record quality for monitoring
        self.monitor.record_quality(symbol, timeframe, quality_report.overall_score)

        # Record gaps for monitoring
        for gap in quality_report.gaps:
            self.monitor.record_gap(
                symbol,
                timeframe,
                datetime.fromisoformat(gap["start"]),
                datetime.fromisoformat(gap["end"]),
                gap["bars_missing"],
            )

        # Reconciliation with secondary sources
        if secondary_data and primary_source:
            clean_df, recon_result = self.reconciler.reconcile(
                clean_df,
                primary_source,
                primary_confidence,
                secondary_data,
                symbol,
            )
            result.discrepancies = len(recon_result.discrepancies)
            result.gaps_filled = recon_result.total_bars_filled
            result.sources_successful = recon_result.sources_used

        result.rows_clean = len(clean_df)

        # Save clean data
        if save_to_db and not clean_df.empty:
            rows_saved = self.storage.save_clean(
                clean_df,
                symbol,
                timeframe,
                quality_report.overall_score,
                result.sources_successful,
            )
            result.rows_stored = rows_saved

        if save_to_parquet and not clean_df.empty:
            self.storage.save_parquet(clean_df, symbol, timeframe)

        result.success = result.rows_clean > 0
        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Completed {symbol} {timeframe}",
            rows=result.rows_clean,
            quality=f"{result.quality_score:.2f}",
            sources=result.sources_successful,
            duration=f"{result.duration_seconds:.1f}s",
        )

        return result

    def run_batch(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parallel: bool = True,
        max_workers: int = 5,
    ) -> Dict[str, Dict[str, ProcessingResult]]:
        """
        Process multiple symbols in batch.

        Args:
            symbols: List of symbols to process.
            timeframes: List of timeframes (default: from config).
            start_date: Start date (default: from config).
            end_date: End date.
            parallel: Process symbols in parallel.
            max_workers: Maximum parallel workers.

        Returns:
            Nested dict: {timeframe: {symbol: ProcessingResult}}.
        """
        timeframes = timeframes or self.config.default_timeframes
        start_date = start_date or self.config.default_start_date

        all_results: Dict[str, Dict[str, ProcessingResult]] = {}
        total_start = time.time()

        for timeframe in timeframes:
            logger.info(f"Processing timeframe: {timeframe}", symbols=len(symbols))
            all_results[timeframe] = {}

            if parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process,
                            symbol,
                            timeframe,
                            start_date,
                            end_date,
                        ): symbol
                        for symbol in symbols
                    }

                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            all_results[timeframe][symbol] = result
                        except Exception as e:
                            logger.error(f"Failed to process {symbol}: {e}")
                            all_results[timeframe][symbol] = ProcessingResult(
                                symbol=symbol,
                                timeframe=timeframe,
                                errors=[str(e)],
                            )
            else:
                for symbol in symbols:
                    result = self.process(symbol, timeframe, start_date, end_date)
                    all_results[timeframe][symbol] = result

        # Cleanup old raw data
        self.storage.cleanup_raw_data()

        total_duration = time.time() - total_start
        logger.info(
            "Batch processing complete",
            timeframes=len(timeframes),
            symbols=len(symbols),
            duration=f"{total_duration:.1f}s",
        )

        return all_results

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "monitoring": self.monitor.get_dashboard_data(),
            "health": self.monitor.get_health_summary(),
            "storage": self.storage.get_parquet_stats(),
            "reconciliation": self.reconciler.get_audit_summary(),
        }

    def get_health(self) -> Dict[str, Any]:
        """Get concise health summary."""
        return self.monitor.get_health_summary()

    def _on_discrepancy_alert(self, discrepancy):
        """Callback for discrepancy alerts."""
        from data.pipeline.monitoring import Alert, AlertLevel

        alert = Alert(
            level=AlertLevel.WARNING,
            title="Data Discrepancy",
            message=f"Price discrepancy of {discrepancy.pct_diff:.4f}% "
                    f"between {discrepancy.source_a} and {discrepancy.source_b}",
            metadata=discrepancy.to_dict(),
        )
        self.monitor.telegram.send(alert)

    def initialize(self) -> None:
        """Initialize pipeline (create tables, etc.)."""
        self.storage.initialize_tables()
        self.monitor.start_background_monitoring()
        logger.info("Pipeline initialized")

    def close(self) -> None:
        """Close pipeline and release resources."""
        self.monitor.stop_background_monitoring()
        self.storage.close()
        logger.info("Pipeline closed")


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Pipeline")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "BTCUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h", "1d"])
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--init", action="store_true", help="Initialize database tables")

    args = parser.parse_args()

    pipeline = DataPipeline()

    try:
        if args.init:
            pipeline.initialize()

        results = pipeline.run_batch(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            parallel=not args.no_parallel,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        for tf, tf_results in results.items():
            print(f"\n{tf}:")
            success = sum(1 for r in tf_results.values() if r.success)
            total_rows = sum(r.rows_clean for r in tf_results.values())
            print(f"  Success: {success}/{len(tf_results)}")
            print(f"  Total rows: {total_rows}")

        # Print health
        health = pipeline.get_health()
        print(f"\nHealth: {health['overall']}")
        print(f"Sources: {health['sources']}")

    finally:
        pipeline.close()
