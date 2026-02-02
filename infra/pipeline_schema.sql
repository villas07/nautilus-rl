-- =============================================================================
-- Data Pipeline Schema for TimescaleDB
-- =============================================================================
--
-- This schema supports the multi-source data pipeline with:
-- - Raw data tables (30-day retention)
-- - Clean data tables (indefinite retention)
-- - Metadata tracking
-- - Audit logging
--
-- Run with: psql -U postgres -d deskgrade -f pipeline_schema.sql
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- RAW DATA TABLES (30-day retention)
-- Stores data as received from sources before quality control
-- =============================================================================

-- Raw 1-minute bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_1min (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(30, 8),
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, source)
);

-- Convert to hypertable
SELECT create_hypertable('raw_ohlcv_1min', 'timestamp', if_not_exists => TRUE);

-- Create index for common queries
CREATE INDEX IF NOT EXISTS idx_raw_1min_symbol ON raw_ohlcv_1min (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_1min_source ON raw_ohlcv_1min (source, timestamp DESC);

-- Retention policy: 30 days
SELECT add_retention_policy('raw_ohlcv_1min', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw 5-minute bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_5min (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_5min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_5min_symbol ON raw_ohlcv_5min (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_5min', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw 15-minute bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_15min (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_15min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_15min_symbol ON raw_ohlcv_15min (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_15min', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw 30-minute bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_30min (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_30min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_30min_symbol ON raw_ohlcv_30min (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_30min', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw 1-hour bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_1hour (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_1hour', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_1hour_symbol ON raw_ohlcv_1hour (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_1hour', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw 4-hour bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_4hour (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_4hour', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_4hour_symbol ON raw_ohlcv_4hour (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_4hour', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw daily bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_1day (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_1day', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_1day_symbol ON raw_ohlcv_1day (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_1day', INTERVAL '30 days', if_not_exists => TRUE);


-- Raw weekly bars
CREATE TABLE IF NOT EXISTS raw_ohlcv_1week (LIKE raw_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('raw_ohlcv_1week', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_1week_symbol ON raw_ohlcv_1week (symbol, timestamp DESC);
SELECT add_retention_policy('raw_ohlcv_1week', INTERVAL '30 days', if_not_exists => TRUE);


-- =============================================================================
-- CLEAN DATA TABLES (indefinite retention)
-- Stores quality-controlled, reconciled data
-- =============================================================================

-- Clean 1-minute bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_1min (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(30, 8),
    quality_score DECIMAL(3, 2) DEFAULT 1.0,
    sources TEXT,  -- Comma-separated list of sources used
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('clean_ohlcv_1min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_1min_symbol ON clean_ohlcv_1min (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_clean_1min_quality ON clean_ohlcv_1min (quality_score) WHERE quality_score < 0.9;


-- Clean 5-minute bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_5min (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_5min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_5min_symbol ON clean_ohlcv_5min (symbol, timestamp DESC);


-- Clean 15-minute bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_15min (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_15min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_15min_symbol ON clean_ohlcv_15min (symbol, timestamp DESC);


-- Clean 30-minute bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_30min (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_30min', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_30min_symbol ON clean_ohlcv_30min (symbol, timestamp DESC);


-- Clean 1-hour bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_1hour (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_1hour', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_1hour_symbol ON clean_ohlcv_1hour (symbol, timestamp DESC);


-- Clean 4-hour bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_4hour (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_4hour', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_4hour_symbol ON clean_ohlcv_4hour (symbol, timestamp DESC);


-- Clean daily bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_1day (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_1day', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_1day_symbol ON clean_ohlcv_1day (symbol, timestamp DESC);


-- Clean weekly bars
CREATE TABLE IF NOT EXISTS clean_ohlcv_1week (LIKE clean_ohlcv_1min INCLUDING ALL);
SELECT create_hypertable('clean_ohlcv_1week', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_1week_symbol ON clean_ohlcv_1week (symbol, timestamp DESC);


-- =============================================================================
-- METADATA TABLES
-- =============================================================================

-- Pipeline metadata
CREATE TABLE IF NOT EXISTS data_pipeline_metadata (
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    source VARCHAR(50) NOT NULL,
    first_bar TIMESTAMPTZ,
    last_bar TIMESTAMPTZ,
    total_bars BIGINT DEFAULT 0,
    quality_score DECIMAL(3, 2),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, source)
);

CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON data_pipeline_metadata (symbol);
CREATE INDEX IF NOT EXISTS idx_metadata_updated ON data_pipeline_metadata (last_updated DESC);


-- Source status tracking
CREATE TABLE IF NOT EXISTS data_source_status (
    source VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'unknown',  -- healthy, degraded, down, unknown
    last_success TIMESTAMPTZ,
    last_failure TIMESTAMPTZ,
    last_error TEXT,
    consecutive_failures INT DEFAULT 0,
    total_requests BIGINT DEFAULT 0,
    total_failures BIGINT DEFAULT 0,
    avg_latency_ms DECIMAL(10, 2),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- =============================================================================
-- AUDIT TABLES
-- =============================================================================

-- Discrepancy audit log
CREATE TABLE IF NOT EXISTS data_discrepancy_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    field VARCHAR(20) NOT NULL,  -- open, high, low, close, volume
    source_a VARCHAR(50) NOT NULL,
    source_b VARCHAR(50) NOT NULL,
    value_a DECIMAL(20, 8),
    value_b DECIMAL(20, 8),
    pct_diff DECIMAL(10, 6),
    level VARCHAR(20),  -- minor, moderate, significant, critical
    resolved_value DECIMAL(20, 8),
    resolved_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('data_discrepancy_log', 'created_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_discrepancy_symbol ON data_discrepancy_log (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_discrepancy_level ON data_discrepancy_log (level, created_at DESC);

-- Retention: 90 days
SELECT add_retention_policy('data_discrepancy_log', INTERVAL '90 days', if_not_exists => TRUE);


-- Gap detection log
CREATE TABLE IF NOT EXISTS data_gap_log (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    gap_start TIMESTAMPTZ NOT NULL,
    gap_end TIMESTAMPTZ NOT NULL,
    bars_missing INT NOT NULL,
    filled BOOLEAN DEFAULT FALSE,
    filled_source VARCHAR(50),
    filled_at TIMESTAMPTZ,
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gap_symbol ON data_gap_log (symbol, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_gap_unfilled ON data_gap_log (filled) WHERE filled = FALSE;


-- Quality report log
CREATE TABLE IF NOT EXISTS data_quality_log (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    timestamp_score DECIMAL(3, 2),
    price_score DECIMAL(3, 2),
    volume_score DECIMAL(3, 2),
    completeness_score DECIMAL(3, 2),
    overall_score DECIMAL(3, 2),
    total_rows INT,
    valid_rows INT,
    issues_count INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('data_quality_log', 'created_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_quality_symbol ON data_quality_log (symbol, created_at DESC);

-- Retention: 365 days
SELECT add_retention_policy('data_quality_log', INTERVAL '365 days', if_not_exists => TRUE);


-- =============================================================================
-- VIEWS
-- =============================================================================

-- Latest data availability per symbol
CREATE OR REPLACE VIEW v_data_availability AS
SELECT
    symbol,
    timeframe,
    source,
    first_bar,
    last_bar,
    total_bars,
    quality_score,
    last_updated,
    EXTRACT(EPOCH FROM (NOW() - last_bar)) / 3600 AS hours_since_last_bar
FROM data_pipeline_metadata
ORDER BY symbol, timeframe;


-- Source health summary
CREATE OR REPLACE VIEW v_source_health AS
SELECT
    source,
    status,
    last_success,
    last_failure,
    consecutive_failures,
    ROUND(100.0 * (total_requests - total_failures) / NULLIF(total_requests, 0), 2) AS success_rate_pct,
    avg_latency_ms
FROM data_source_status
ORDER BY
    CASE status
        WHEN 'down' THEN 1
        WHEN 'degraded' THEN 2
        WHEN 'healthy' THEN 3
        ELSE 4
    END;


-- Recent discrepancies summary
CREATE OR REPLACE VIEW v_recent_discrepancies AS
SELECT
    DATE_TRUNC('hour', created_at) AS hour,
    symbol,
    level,
    COUNT(*) AS count,
    AVG(pct_diff) AS avg_pct_diff
FROM data_discrepancy_log
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY 1, 2, 3
ORDER BY 1 DESC, count DESC;


-- Unfilled gaps
CREATE OR REPLACE VIEW v_unfilled_gaps AS
SELECT
    symbol,
    timeframe,
    gap_start,
    gap_end,
    bars_missing,
    detected_at
FROM data_gap_log
WHERE filled = FALSE
ORDER BY bars_missing DESC;


-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to get data coverage for a symbol
CREATE OR REPLACE FUNCTION get_data_coverage(
    p_symbol VARCHAR,
    p_timeframe VARCHAR
)
RETURNS TABLE (
    date_range TSTZRANGE,
    total_bars BIGINT,
    expected_bars BIGINT,
    coverage_pct DECIMAL
) AS $$
DECLARE
    v_table_name VARCHAR;
    v_first_bar TIMESTAMPTZ;
    v_last_bar TIMESTAMPTZ;
    v_total BIGINT;
    v_expected BIGINT;
    v_interval INTERVAL;
BEGIN
    v_table_name := 'clean_ohlcv_' ||
        CASE p_timeframe
            WHEN '1m' THEN '1min'
            WHEN '5m' THEN '5min'
            WHEN '15m' THEN '15min'
            WHEN '30m' THEN '30min'
            WHEN '1h' THEN '1hour'
            WHEN '4h' THEN '4hour'
            WHEN '1d' THEN '1day'
            WHEN '1w' THEN '1week'
            ELSE p_timeframe
        END;

    v_interval := CASE p_timeframe
        WHEN '1m' THEN INTERVAL '1 minute'
        WHEN '5m' THEN INTERVAL '5 minutes'
        WHEN '15m' THEN INTERVAL '15 minutes'
        WHEN '30m' THEN INTERVAL '30 minutes'
        WHEN '1h' THEN INTERVAL '1 hour'
        WHEN '4h' THEN INTERVAL '4 hours'
        WHEN '1d' THEN INTERVAL '1 day'
        WHEN '1w' THEN INTERVAL '1 week'
        ELSE INTERVAL '1 hour'
    END;

    EXECUTE format('
        SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
        FROM %I
        WHERE symbol = $1
    ', v_table_name)
    INTO v_first_bar, v_last_bar, v_total
    USING p_symbol;

    IF v_first_bar IS NOT NULL AND v_last_bar IS NOT NULL THEN
        v_expected := EXTRACT(EPOCH FROM (v_last_bar - v_first_bar)) / EXTRACT(EPOCH FROM v_interval) + 1;
    ELSE
        v_expected := 0;
    END IF;

    RETURN QUERY SELECT
        TSTZRANGE(v_first_bar, v_last_bar, '[]'),
        v_total,
        v_expected,
        CASE WHEN v_expected > 0 THEN ROUND(100.0 * v_total / v_expected, 2) ELSE 0 END;
END;
$$ LANGUAGE plpgsql;


-- Function to log a discrepancy
CREATE OR REPLACE FUNCTION log_discrepancy(
    p_timestamp TIMESTAMPTZ,
    p_symbol VARCHAR,
    p_field VARCHAR,
    p_source_a VARCHAR,
    p_source_b VARCHAR,
    p_value_a DECIMAL,
    p_value_b DECIMAL,
    p_resolved_value DECIMAL,
    p_resolved_source VARCHAR
) RETURNS VOID AS $$
DECLARE
    v_pct_diff DECIMAL;
    v_level VARCHAR;
BEGIN
    -- Calculate percentage difference
    IF p_value_a != 0 THEN
        v_pct_diff := ABS(p_value_a - p_value_b) / ABS(p_value_a) * 100;
    ELSE
        v_pct_diff := CASE WHEN p_value_b = 0 THEN 0 ELSE 100 END;
    END IF;

    -- Determine level
    v_level := CASE
        WHEN v_pct_diff < 0.01 THEN 'minor'
        WHEN v_pct_diff < 0.1 THEN 'moderate'
        WHEN v_pct_diff < 1.0 THEN 'significant'
        ELSE 'critical'
    END;

    INSERT INTO data_discrepancy_log (
        timestamp, symbol, field, source_a, source_b,
        value_a, value_b, pct_diff, level,
        resolved_value, resolved_source
    ) VALUES (
        p_timestamp, p_symbol, p_field, p_source_a, p_source_b,
        p_value_a, p_value_b, v_pct_diff, v_level,
        p_resolved_value, p_resolved_source
    );
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- GRANTS (adjust as needed)
-- =============================================================================

-- Grant read access to monitoring user (if exists)
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring_user;

-- Grant full access to application user (if exists)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO nautilus_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nautilus_app;


-- =============================================================================
-- DONE
-- =============================================================================
SELECT 'Pipeline schema created successfully' AS status;
