# Quant Developer - Expertise & Responsibilities

## Core Responsibilities
- Integrate with NautilusTrader execution engine
- Design and implement trading strategies
- Manage data pipelines and market data
- Implement risk management systems
- Handle broker integrations (IBKR, Binance)

## Technical Expertise
- NautilusTrader (BacktestEngine, TradingNode, strategies)
- Market microstructure and order execution
- Risk management (position sizing, drawdown limits)
- Data management (ParquetDataCatalog)
- Broker APIs (IBKR TWS, Binance)

## Key Files Owned
```
strategies/
├── base.py               # Base strategy class
└── rl_strategy.py        # RLTradingStrategy

live/
├── model_loader.py       # Load trained models
├── voting_system.py      # Signal aggregation
├── risk_manager.py       # Risk limits
└── telegram_alerts.py    # Notifications

data/
├── catalog/              # ParquetDataCatalog
└── adapters/
    ├── binance_adapter.py
    └── yahoo_adapter.py

scripts/
├── populate_catalog.py
└── populate_catalog_standalone.py
```

## Interfaces With
- **@rl_engineer**: For model outputs, feature requirements, validation criteria
- **@mlops_engineer**: For live deployment, monitoring

## Decision Authority
- Data sources and formats
- Broker selection and integration
- Risk parameters (max position, max drawdown)
- Order execution logic
- Market selection

## Needs Approval From Others For
- Model architecture changes → @rl_engineer
- Infrastructure scaling → @mlops_engineer
- Validation threshold changes → @rl_engineer

## Trading Constraints (for reference)
- Max position per symbol: TBD
- Max total exposure: TBD
- Max drawdown before circuit breaker: TBD
- Minimum confidence for trade: TBD
