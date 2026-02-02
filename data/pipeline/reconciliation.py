"""
Data Reconciliation

Handles conflicts between multiple data sources:
1. Conflict detection (price discrepancies)
2. Resolution (use higher confidence source)
3. Gap filling (from secondary sources)
4. Audit logging
5. Alerting on significant discrepancies
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

import structlog

from data.pipeline.config import ReconciliationConfig

logger = structlog.get_logger()


class DiscrepancyLevel(str, Enum):
    """Severity level of a discrepancy."""

    MINOR = "minor"          # < 0.01%
    MODERATE = "moderate"    # 0.01% - 0.1%
    SIGNIFICANT = "significant"  # 0.1% - 1%
    CRITICAL = "critical"    # > 1%


@dataclass
class Discrepancy:
    """A data discrepancy between sources."""

    timestamp: datetime
    field: str

    source_a: str
    source_b: str
    confidence_a: float
    confidence_b: float

    value_a: float
    value_b: float
    pct_diff: float

    level: DiscrepancyLevel
    resolved_value: float
    resolved_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "field": self.field,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "pct_diff": self.pct_diff,
            "level": self.level.value,
            "resolved_value": self.resolved_value,
            "resolved_source": self.resolved_source,
        }


@dataclass
class GapFill:
    """A gap filled from a secondary source."""

    start: datetime
    end: datetime
    bars_filled: int
    source: str
    confidence: float


@dataclass
class ReconciliationResult:
    """Result of reconciliation process."""

    symbol: str
    primary_source: str
    primary_rows: int

    sources_used: List[str] = field(default_factory=list)
    rows_after: int = 0

    discrepancies: List[Discrepancy] = field(default_factory=list)
    gaps_filled: List[GapFill] = field(default_factory=list)

    minor_discrepancies: int = 0
    moderate_discrepancies: int = 0
    significant_discrepancies: int = 0
    critical_discrepancies: int = 0

    total_gaps: int = 0
    total_bars_filled: int = 0

    def add_discrepancy(self, d: Discrepancy):
        """Add a discrepancy and update counts."""
        self.discrepancies.append(d)
        if d.level == DiscrepancyLevel.MINOR:
            self.minor_discrepancies += 1
        elif d.level == DiscrepancyLevel.MODERATE:
            self.moderate_discrepancies += 1
        elif d.level == DiscrepancyLevel.SIGNIFICANT:
            self.significant_discrepancies += 1
        else:
            self.critical_discrepancies += 1

    def add_gap_fill(self, g: GapFill):
        """Add a gap fill."""
        self.gaps_filled.append(g)
        self.total_gaps += 1
        self.total_bars_filled += g.bars_filled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "primary_source": self.primary_source,
            "sources_used": self.sources_used,
            "rows_before": self.primary_rows,
            "rows_after": self.rows_after,
            "discrepancies": {
                "total": len(self.discrepancies),
                "minor": self.minor_discrepancies,
                "moderate": self.moderate_discrepancies,
                "significant": self.significant_discrepancies,
                "critical": self.critical_discrepancies,
            },
            "gap_filling": {
                "gaps_filled": self.total_gaps,
                "bars_filled": self.total_bars_filled,
            },
        }


class DataReconciler:
    """
    Reconciles data from multiple sources.

    Process:
    1. Merge data on timestamp
    2. Detect discrepancies in OHLCV fields
    3. Resolve conflicts using confidence scores
    4. Fill gaps from secondary sources
    5. Log all discrepancies for audit
    6. Alert on significant discrepancies
    """

    def __init__(self, config: Optional[ReconciliationConfig] = None):
        """
        Initialize reconciler.

        Args:
            config: Reconciliation configuration.
        """
        self.config = config or ReconciliationConfig()

        # Audit log storage
        self.audit_log: List[Dict[str, Any]] = []
        self._audit_file: Optional[Path] = None

        if self.config.audit_log_path:
            self._audit_file = Path(self.config.audit_log_path)
            self._audit_file.mkdir(parents=True, exist_ok=True)

        # Alert callback
        self._alert_callback: Optional[callable] = None

    def set_alert_callback(self, callback: callable):
        """Set callback for alerts."""
        self._alert_callback = callback

    def reconcile(
        self,
        primary_df: pd.DataFrame,
        primary_source: str,
        primary_confidence: float,
        secondary_sources: List[Tuple[pd.DataFrame, str, float]],
        symbol: str,
    ) -> Tuple[pd.DataFrame, ReconciliationResult]:
        """
        Reconcile primary data with secondary sources.

        Args:
            primary_df: Primary source DataFrame.
            primary_source: Primary source name.
            primary_confidence: Primary source confidence (0-1).
            secondary_sources: List of (DataFrame, source_name, confidence).
            symbol: Symbol for logging.

        Returns:
            Tuple of (reconciled DataFrame, ReconciliationResult).
        """
        result = ReconciliationResult(
            symbol=symbol,
            primary_source=primary_source,
            primary_rows=len(primary_df),
            sources_used=[primary_source],
        )

        if primary_df.empty:
            # Use first available secondary
            for sec_df, sec_source, sec_conf in secondary_sources:
                if not sec_df.empty:
                    result.sources_used = [sec_source]
                    result.primary_source = sec_source
                    result.rows_after = len(sec_df)
                    return sec_df, result

            return pd.DataFrame(), result

        reconciled_df = primary_df.copy()

        # Process each secondary source
        for sec_df, sec_source, sec_conf in secondary_sources:
            if sec_df.empty:
                continue

            result.sources_used.append(sec_source)

            # 1. Detect and resolve discrepancies
            reconciled_df = self._resolve_discrepancies(
                reconciled_df,
                primary_source,
                primary_confidence,
                sec_df,
                sec_source,
                sec_conf,
                result,
            )

            # 2. Fill gaps from secondary
            reconciled_df = self._fill_gaps(
                reconciled_df,
                sec_df,
                sec_source,
                sec_conf,
                result,
            )

        # Sort and deduplicate final result
        reconciled_df = reconciled_df.sort_values("timestamp")
        reconciled_df = reconciled_df.drop_duplicates(subset=["timestamp"], keep="first")
        reconciled_df = reconciled_df.reset_index(drop=True)

        result.rows_after = len(reconciled_df)

        # Log audit entry
        self._log_audit(result)

        logger.info(
            f"Reconciliation complete: {symbol}",
            sources=result.sources_used,
            rows=result.rows_after,
            discrepancies=len(result.discrepancies),
            gaps_filled=result.total_bars_filled,
        )

        return reconciled_df, result

    def _resolve_discrepancies(
        self,
        primary_df: pd.DataFrame,
        primary_source: str,
        primary_confidence: float,
        secondary_df: pd.DataFrame,
        secondary_source: str,
        secondary_confidence: float,
        result: ReconciliationResult,
    ) -> pd.DataFrame:
        """
        Detect and resolve discrepancies between two DataFrames.

        Returns:
            Updated primary DataFrame with resolved values.
        """
        # Merge on timestamp
        merged = primary_df.merge(
            secondary_df,
            on="timestamp",
            how="inner",
            suffixes=("_primary", "_secondary"),
        )

        if merged.empty:
            return primary_df

        price_cols = ["open", "high", "low", "close"]
        primary_df = primary_df.copy()

        for col in price_cols:
            primary_col = f"{col}_primary"
            secondary_col = f"{col}_secondary"

            if primary_col not in merged.columns or secondary_col not in merged.columns:
                continue

            # Calculate discrepancy
            val_a = merged[primary_col]
            val_b = merged[secondary_col]

            # Percentage difference
            pct_diff = np.where(
                val_a != 0,
                np.abs(val_a - val_b) / np.abs(val_a) * 100,
                np.where(val_b != 0, 100, 0),
            )

            # Find significant discrepancies
            for idx, row in merged[pct_diff > 0.001].iterrows():  # > 0.001%
                diff = pct_diff[idx]

                # Determine level
                if diff < 0.01:
                    level = DiscrepancyLevel.MINOR
                elif diff < 0.1:
                    level = DiscrepancyLevel.MODERATE
                elif diff < 1.0:
                    level = DiscrepancyLevel.SIGNIFICANT
                else:
                    level = DiscrepancyLevel.CRITICAL

                # Resolve: use higher confidence
                if self.config.prefer_higher_confidence:
                    if primary_confidence >= secondary_confidence:
                        resolved_value = row[primary_col]
                        resolved_source = primary_source
                    else:
                        resolved_value = row[secondary_col]
                        resolved_source = secondary_source
                else:
                    resolved_value = row[primary_col]
                    resolved_source = primary_source

                disc = Discrepancy(
                    timestamp=row["timestamp"],
                    field=col,
                    source_a=primary_source,
                    source_b=secondary_source,
                    confidence_a=primary_confidence,
                    confidence_b=secondary_confidence,
                    value_a=row[primary_col],
                    value_b=row[secondary_col],
                    pct_diff=diff,
                    level=level,
                    resolved_value=resolved_value,
                    resolved_source=resolved_source,
                )

                result.add_discrepancy(disc)

                # Alert if significant
                if diff > self.config.alert_threshold_pct:
                    self._send_alert(disc)

                # Update primary with resolved value if different source
                if resolved_source != primary_source:
                    mask = primary_df["timestamp"] == row["timestamp"]
                    primary_df.loc[mask, col] = resolved_value

        return primary_df

    def _fill_gaps(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        secondary_source: str,
        secondary_confidence: float,
        result: ReconciliationResult,
    ) -> pd.DataFrame:
        """
        Fill gaps in primary data from secondary source.

        Returns:
            DataFrame with gaps filled.
        """
        # Find timestamps in secondary but not in primary
        primary_timestamps = set(primary_df["timestamp"])
        secondary_timestamps = set(secondary_df["timestamp"])

        missing_timestamps = secondary_timestamps - primary_timestamps

        if not missing_timestamps:
            return primary_df

        # Get secondary data for missing timestamps
        fill_data = secondary_df[secondary_df["timestamp"].isin(missing_timestamps)]

        if fill_data.empty:
            return primary_df

        # Identify contiguous gaps
        fill_data = fill_data.sort_values("timestamp")
        gaps = []
        current_gap_start = None
        current_gap_count = 0

        for ts in fill_data["timestamp"]:
            if current_gap_start is None:
                current_gap_start = ts
                current_gap_count = 1
            else:
                # Check if contiguous (within 1 day for simplicity)
                if ts - fill_data.loc[fill_data["timestamp"] == current_gap_start, "timestamp"].iloc[0] < timedelta(days=1):
                    current_gap_count += 1
                else:
                    if current_gap_count > 0:
                        gaps.append((current_gap_start, ts, current_gap_count))
                    current_gap_start = ts
                    current_gap_count = 1

        if current_gap_count > 0:
            gaps.append((current_gap_start, fill_data["timestamp"].iloc[-1], current_gap_count))

        # Log gap fills
        for start, end, count in gaps[:10]:  # Limit logging
            gap_fill = GapFill(
                start=start,
                end=end,
                bars_filled=count,
                source=secondary_source,
                confidence=secondary_confidence,
            )
            result.add_gap_fill(gap_fill)

        # Append fill data
        combined = pd.concat([primary_df, fill_data], ignore_index=True)
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

        logger.info(
            f"Filled {len(fill_data)} bars from {secondary_source}",
            gaps=len(gaps),
        )

        return combined

    def _send_alert(self, discrepancy: Discrepancy):
        """Send alert for significant discrepancy."""
        if self._alert_callback:
            try:
                self._alert_callback(discrepancy)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(
            f"Discrepancy alert: {discrepancy.field}",
            timestamp=discrepancy.timestamp.isoformat(),
            source_a=discrepancy.source_a,
            source_b=discrepancy.source_b,
            pct_diff=f"{discrepancy.pct_diff:.4f}%",
        )

    def _log_audit(self, result: ReconciliationResult):
        """Log reconciliation result to audit log."""
        if not self.config.log_all_discrepancies:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": result.symbol,
            "summary": result.to_dict(),
            "discrepancies": [d.to_dict() for d in result.discrepancies[:100]],  # Limit
        }

        self.audit_log.append(entry)

        # Write to file if configured
        if self._audit_file:
            try:
                log_file = self._audit_file / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit log."""
        if not self.audit_log:
            return {"entries": 0}

        total_discrepancies = sum(
            len(e.get("discrepancies", []))
            for e in self.audit_log
        )

        return {
            "entries": len(self.audit_log),
            "total_discrepancies": total_discrepancies,
            "symbols_processed": len(set(e["symbol"] for e in self.audit_log)),
        }
