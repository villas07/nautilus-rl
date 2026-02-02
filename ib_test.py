#!/usr/bin/env python3
"""
Simple IB Connection Test

Run this directly to test Interactive Brokers connectivity:
    python ib_test.py
"""

import os
import signal
from dotenv import load_dotenv

load_dotenv()

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


def main():
    """Run simple IB connection test."""
    print("=" * 60)
    print("NAUTILUS TRADER - IB CONNECTION TEST")
    print("=" * 60)

    # Get config from environment
    ibkr_host = os.getenv("IBKR_HOST", "host.docker.internal")
    ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
    ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
    ibkr_account = os.getenv("IBKR_ACCOUNT", "")

    print(f"Connecting to IB Gateway at {ibkr_host}:{ibkr_port}")
    print(f"Client ID: {ibkr_client_id}, Account: {ibkr_account}")

    # Create configs
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

    # Create and configure node
    node = TradingNode(config=node_config)
    node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
    node.add_exec_client_factory("IB", InteractiveBrokersLiveExecClientFactory)

    print("Building trading node...")
    node.build()

    print("Starting trading node...")
    print("Press Ctrl+C to stop")

    # Handle shutdown
    def shutdown_handler(signum, frame):
        print("\nShutdown signal received")
        node.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        node.run()
    finally:
        print("Disposing node...")
        node.dispose()
        print("Done")


if __name__ == "__main__":
    main()
