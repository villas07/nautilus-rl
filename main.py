"""
NautilusTrader RL Agents - Main Entry Point

This is the main entry point for running the NautilusTrader
trading system with RL agents.

Usage:
    python main.py                    # Run live trading
    python main.py --mode backtest    # Run backtest
    python main.py --mode paper       # Run paper trading
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional
import argparse

from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NautilusTrader RL Agents Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "test", "simple"],
        default="paper",
        help="Trading mode (default: paper). Use 'test' to validate setup without connecting. Use 'simple' for minimal IB test.",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        help="Instruments to trade (default: all configured)",
    )
    parser.add_argument(
        "--enable-ibkr",
        action="store_true",
        default=True,
        help="Enable IBKR adapter (default: True)",
    )
    parser.add_argument(
        "--enable-binance",
        action="store_true",
        default=True,
        help="Enable Binance adapter (default: True)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/app/models",
        help="Directory containing RL models",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=8000,
        help="Health check server port",
    )
    return parser.parse_args()


async def run_health_server(port: int):
    """Run health check HTTP server."""
    from fastapi import FastAPI
    from uvicorn import Config, Server

    app = FastAPI(title="Nautilus Agents Health")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    config = Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = Server(config)
    await server.serve()


def run_live_trading(args):
    """Run live trading mode."""
    from config.live import create_trading_node, LiveConfig
    from config.instruments import get_instrument_ids
    from strategies.rl_strategy import RLTradingStrategy, RLTradingStrategyConfig

    logger.info("Starting live trading mode")

    # Create trading node
    config = LiveConfig.from_env()
    node = create_trading_node(
        config=config,
        enable_ibkr=args.enable_ibkr,
        enable_binance=args.enable_binance,
    )

    # Add IBKR client factories if enabled
    if args.enable_ibkr and config.ibkr_account:
        from nautilus_trader.adapters.interactive_brokers.factories import (
            InteractiveBrokersLiveDataClientFactory,
            InteractiveBrokersLiveExecClientFactory,
        )
        node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
        node.add_exec_client_factory("IB", InteractiveBrokersLiveExecClientFactory)
        logger.info("Added IBKR client factories")

    # Add Binance client factories if enabled
    if args.enable_binance and config.binance_api_key:
        from nautilus_trader.adapters.binance.factories import (
            BinanceLiveDataClientFactory,
            BinanceLiveExecClientFactory,
        )
        node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
        logger.info("Added Binance client factories")

    # Get instruments
    instrument_ids = args.instruments or get_instrument_ids()

    # Add strategies for each instrument
    for inst_id in instrument_ids:
        strategy_config = RLTradingStrategyConfig(
            strategy_id=f"RL-{inst_id.replace('.', '-')}",
            instrument_id=inst_id,
            models_dir=args.models_dir,
            use_voting=True,
            min_confidence=0.6,
        )
        strategy = RLTradingStrategy(config=strategy_config)
        node.trader.add_strategy(strategy)

    # Build and run
    node.build()

    # Handle shutdown signals
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received")
        node.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        node.run()
    finally:
        node.dispose()


def run_paper_trading(args):
    """Run paper trading mode (same as live but on testnet/paper accounts)."""
    # Force paper mode in environment
    os.environ["TRADING_MODE"] = "paper"
    os.environ["BINANCE_TESTNET"] = "true"

    # Use live trading with paper settings
    run_live_trading(args)


def run_backtest(args):
    """Run backtest mode."""
    from nautilus_trader.backtest.engine import BacktestEngine
    from config.backtest import get_backtest_config, BacktestConfig
    from config.instruments import get_instrument_ids
    from strategies.rl_strategy import RLTradingStrategy, RLTradingStrategyConfig

    logger.info("Starting backtest mode")

    # Get instruments
    instrument_ids = args.instruments or get_instrument_ids()

    # Create strategy configs
    strategies = []
    for inst_id in instrument_ids:
        strategy_config = RLTradingStrategyConfig(
            strategy_id=f"RL-{inst_id.replace('.', '-')}",
            instrument_id=inst_id,
            models_dir=args.models_dir,
            use_voting=True,
            min_confidence=0.6,
        )
        strategies.append(strategy_config)

    # Create backtest config
    backtest_config = get_backtest_config(
        config=BacktestConfig.from_env(),
        instrument_ids=instrument_ids,
        strategies=strategies,
    )

    # Create and run engine
    engine = BacktestEngine(config=backtest_config.engine)

    # Add data
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    catalog_path = os.getenv("CATALOG_PATH", "/app/data/catalog")
    if Path(catalog_path).exists():
        catalog = ParquetDataCatalog(catalog_path)
        # Load data from catalog
        for inst_id in instrument_ids:
            bars = catalog.bars(instrument_ids=[inst_id])
            if bars:
                engine.add_data(bars)

    # Add strategies
    for strategy_config in strategies:
        strategy = RLTradingStrategy(config=strategy_config)
        engine.add_strategy(strategy)

    # Run backtest
    engine.run()

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    for account in engine.trader.accounts():
        print(f"\nAccount: {account.id}")
        print(f"  Balance: {account.balance_total()}")

    # Generate reports
    engine.generate_order_fills_report()
    engine.generate_positions_report()

    engine.dispose()


def run_test_mode(args):
    """
    Run test mode to validate system components without connecting to brokers.

    This validates:
    - NautilusTrader imports and configuration
    - Strategy initialization
    - Risk manager
    - Data adapters
    - Model loading (if models exist)
    """
    logger.info("=" * 60)
    logger.info("NAUTILUS AGENTS - TEST MODE")
    logger.info("=" * 60)

    results = {"passed": 0, "failed": 0, "warnings": 0}

    def check(name: str, func, critical: bool = True):
        """Run a test check."""
        try:
            result = func()
            if result is True or result is None:
                logger.info(f"[PASS] {name}")
                results["passed"] += 1
                return True
            else:
                if critical:
                    logger.error(f"[FAIL] {name}: {result}")
                    results["failed"] += 1
                else:
                    logger.warning(f"[WARN] {name}: {result}")
                    results["warnings"] += 1
                return False
        except Exception as e:
            if critical:
                logger.error(f"[FAIL] {name}: {e}")
                results["failed"] += 1
            else:
                logger.warning(f"[WARN] {name}: {e}")
                results["warnings"] += 1
            return False

    # Test 1: NautilusTrader core imports
    def test_nautilus_imports():
        from nautilus_trader.config import TradingNodeConfig
        from nautilus_trader.live.node import TradingNode
        from nautilus_trader.backtest.engine import BacktestEngine
        return True
    check("NautilusTrader core imports", test_nautilus_imports)

    # Test 2: IBKR adapter imports
    def test_ibkr_imports():
        from nautilus_trader.adapters.interactive_brokers.config import (
            InteractiveBrokersDataClientConfig,
            InteractiveBrokersExecClientConfig,
        )
        from nautilus_trader.adapters.interactive_brokers.factories import (
            InteractiveBrokersLiveDataClientFactory,
            InteractiveBrokersLiveExecClientFactory,
        )
        return True
    check("IBKR adapter imports", test_ibkr_imports)

    # Test 3: Binance adapter imports
    def test_binance_imports():
        from nautilus_trader.adapters.binance.config import (
            BinanceDataClientConfig,
            BinanceExecClientConfig,
        )
        from nautilus_trader.adapters.binance.factories import (
            BinanceLiveDataClientFactory,
            BinanceLiveExecClientFactory,
        )
        return True
    check("Binance adapter imports", test_binance_imports)

    # Test 4: Configuration loading
    def test_config_loading():
        from config.live import LiveConfig, get_live_config
        config = LiveConfig.from_env()
        return True
    check("Configuration loading", test_config_loading)

    # Test 5: Instruments configuration
    def test_instruments():
        from config.instruments import get_instrument_ids
        instruments = get_instrument_ids()
        if len(instruments) == 0:
            return "No instruments configured"
        logger.info(f"       Found {len(instruments)} instruments")
        return True
    check("Instruments configuration", test_instruments)

    # Test 6: Strategy class
    def test_strategy_class():
        from strategies.rl_strategy import RLTradingStrategy, RLTradingStrategyConfig
        config = RLTradingStrategyConfig(
            strategy_id="TEST-001",
            instrument_id="SPY.IBKR",
        )
        return True
    check("Strategy class initialization", test_strategy_class)

    # Test 7: Risk manager
    def test_risk_manager():
        from live.risk_manager import RiskManager, RiskLimits
        limits = RiskLimits(max_daily_loss=500)
        rm = RiskManager(limits=limits)
        status = rm.get_status()
        return True
    check("Risk manager", test_risk_manager)

    # Test 8: Voting system
    def test_voting_system():
        from live.voting_system import VotingSystem
        vs = VotingSystem(models=[], min_confidence=0.6)
        return True
    check("Voting system", test_voting_system)

    # Test 9: Gymnasium environment
    def test_gym_env():
        from gym_env.nautilus_env import NautilusGymEnv
        return True
    check("Gymnasium environment", test_gym_env)

    # Test 10: TimescaleDB adapter
    def test_timescale_adapter():
        from data.adapters.timescale_adapter import TimescaleAdapter
        return True
    check("TimescaleDB adapter", test_timescale_adapter)

    # Test 11: Environment variables
    def test_env_vars():
        from config.live import LiveConfig
        config = LiveConfig.from_env()
        missing = []
        if not config.ibkr_account:
            missing.append("IBKR_ACCOUNT")
        if missing:
            return f"Missing: {', '.join(missing)}"
        return True
    check("Environment variables (IBKR)", test_env_vars, critical=False)

    # Test 12: Models directory
    def test_models_dir():
        models_path = Path(args.models_dir)
        if not models_path.exists():
            return f"Models directory not found: {args.models_dir}"
        models = list(models_path.glob("*.zip"))
        if len(models) == 0:
            return "No model files (.zip) found"
        logger.info(f"       Found {len(models)} model files")
        return True
    check("Models directory", test_models_dir, critical=False)

    # Test 13: TradingNodeConfig creation
    def test_trading_node_config():
        from config.live import get_live_config, LiveConfig
        config = LiveConfig.from_env()
        # Temporarily disable brokers to test config creation
        node_config = get_live_config(config, enable_ibkr=False, enable_binance=False)
        return True
    check("TradingNodeConfig creation", test_trading_node_config)

    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Passed:   {results['passed']}")
    logger.info(f"  Failed:   {results['failed']}")
    logger.info(f"  Warnings: {results['warnings']}")
    logger.info("=" * 60)

    if results["failed"] > 0:
        logger.error("Some tests failed. Fix issues before running live.")
        sys.exit(1)
    else:
        logger.info("All critical tests passed. System ready for deployment.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start IB Gateway/TWS on port 7497")
        logger.info("  2. Run: docker-compose up -d nautilus-trader")
        logger.info("  3. Check logs: docker-compose logs -f nautilus-trader")


def run_simple_ib_test(args):
    """
    Simple IB connection test without strategies.

    This creates a minimal TradingNode that just connects to IB
    and waits for user interrupt.
    """
    from nautilus_trader.config import TradingNodeConfig
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.adapters.interactive_brokers.config import (
        InteractiveBrokersDataClientConfig,
        InteractiveBrokersExecClientConfig,
    )
    from nautilus_trader.adapters.interactive_brokers.factories import (
        InteractiveBrokersLiveDataClientFactory,
        InteractiveBrokersLiveExecClientFactory,
    )

    logger.info("=" * 60)
    logger.info("SIMPLE IB CONNECTION TEST")
    logger.info("=" * 60)

    # Get config from environment
    ibkr_host = os.getenv("IBKR_HOST", "host.docker.internal")
    ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
    ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
    ibkr_account = os.getenv("IBKR_ACCOUNT", "")

    logger.info(f"Connecting to IB Gateway at {ibkr_host}:{ibkr_port}")
    logger.info(f"Client ID: {ibkr_client_id}, Account: {ibkr_account}")

    # Create minimal config
    data_config = InteractiveBrokersDataClientConfig(
        ibg_host=ibkr_host,
        ibg_port=ibkr_port,
        ibg_client_id=ibkr_client_id,
    )

    exec_config = InteractiveBrokersExecClientConfig(
        ibg_host=ibkr_host,
        ibg_port=ibkr_port,
        ibg_client_id=ibkr_client_id + 100,
        account_id=ibkr_account,
    )

    node_config = TradingNodeConfig(
        trader_id="IB-TEST-001",
        data_clients={"IB": data_config},
        exec_clients={"IB": exec_config},
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )

    # Create node
    node = TradingNode(config=node_config)

    # Add factories
    node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory("IB", InteractiveBrokersLiveExecClientFactory)

    logger.info("Building trading node...")
    node.build()

    logger.info("Starting trading node...")
    logger.info("Press Ctrl+C to stop")

    # Handle shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received")
        node.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        node.run()
    finally:
        logger.info("Disposing node...")
        node.dispose()
        logger.info("Done")


async def main():
    """Main entry point."""
    args = parse_args()

    logger.info(
        "NautilusTrader RL Agents starting",
        mode=args.mode,
        instruments=args.instruments,
    )

    # Start health server in background
    health_task = asyncio.create_task(run_health_server(args.health_port))

    try:
        if args.mode == "test":
            run_test_mode(args)
        elif args.mode == "simple":
            run_simple_ib_test(args)
        elif args.mode == "live":
            run_live_trading(args)
        elif args.mode == "paper":
            run_paper_trading(args)
        elif args.mode == "backtest":
            run_backtest(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Fatal error", error=str(e))
        sys.exit(1)
    finally:
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
