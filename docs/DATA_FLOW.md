# Data Flow Architecture

## Gymnasium Environment

El entorno `NautilusGymEnv` usa integración nativa con NautilusTrader:

| Componente | Descripción |
|------------|-------------|
| **ParquetDataCatalog** | Carga datos desde `/app/data/catalog` |
| **BacktestEngine** | Ejecución realista con slippage |
| **Order Matching** | Matching engine de NautilusTrader |

```python
from gym_env import NautilusGymEnv, NautilusEnvConfig

config = NautilusEnvConfig(
    instrument_id="BTCUSDT.BINANCE",
    catalog_path="/app/data/catalog",
)
env = NautilusGymEnv(config)
```

---

## Complete Data Flow: Sources → Agents

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL DATA SOURCES                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   HISTORICAL DATA                              REAL-TIME DATA                           │
│   ═══════════════                              ══════════════                           │
│                                                                                         │
│   ┌──────────────┐  ┌──────────────┐          ┌──────────────┐                         │
│   │  DATABENTO   │  │   BINANCE    │          │     IBKR     │                         │
│   │  (Stocks)    │  │   (Crypto)   │          │   Gateway    │                         │
│   │  Premium     │  │    FREE      │          │  TWS 10.43   │                         │
│   │  $$$         │  │   ∞ calls    │          │  Port 7497   │                         │
│   └──────┬───────┘  └──────┬───────┘          └──────┬───────┘                         │
│          │                 │                         │                                  │
│   ┌──────┴───────┐  ┌──────┴───────┐                │                                  │
│   │   POLYGON    │  │ CRYPTOCOMPARE│                │                                  │
│   │  (Backup)    │  │   (Backup)   │                │                                  │
│   │  $28/mo      │  │  100k/month  │                │                                  │
│   └──────┬───────┘  └──────┬───────┘                │                                  │
│          │                 │                         │                                  │
│   ┌──────┴───────┐  ┌──────┴───────┐                │                                  │
│   │    YAHOO     │  │  COINGECKO   │                │                                  │
│   │  (Fallback)  │  │  (Fallback)  │                │                                  │
│   │    FREE      │  │    FREE      │                │                                  │
│   └──────┬───────┘  └──────┬───────┘                │                                  │
│          │                 │                         │                                  │
│   ┌──────┴───────┐                                  │                                  │
│   │     FRED     │  ◄── Macro indicators           │                                  │
│   │    FREE      │      (GDP, CPI, rates)          │                                  │
│   └──────┬───────┘                                  │                                  │
│          │                                          │                                  │
└──────────┼──────────────────────────────────────────┼──────────────────────────────────┘
           │                                          │
           ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                               │
│                         (data/pipeline/manager.py)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           1. INGESTION LAYER                                    │   │
│   │                                                                                 │   │
│   │   Adapters:                              Priority Logic:                        │   │
│   │   • BinanceHistoricalAdapter             • Stocks: Databento → Polygon → Yahoo │   │
│   │   • YahooFinanceAdapter                  • Crypto: Binance → CryptoCompare     │   │
│   │   • CryptoCompareAdapter                 • Macro: FRED                         │   │
│   │   • AlphaVantageAdapter                                                        │   │
│   │   • FREDAdapter                                                                │   │
│   │                                                                                 │   │
│   └─────────────────────────────────┬───────────────────────────────────────────────┘   │
│                                     │                                                   │
│                                     ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         2. QUALITY CONTROL                                      │   │
│   │                       (data/pipeline/quality.py)                                │   │
│   │                                                                                 │   │
│   │   Checks:                                 Actions:                              │   │
│   │   • Timestamp validation (market hours)  • Flag invalid data                   │   │
│   │   • Outlier detection (>3σ)              • Score: 0.0 - 1.0                    │   │
│   │   • Volume validation (zero check)       • Generate QualityReport              │   │
│   │   • OHLC sanity (high >= low)            • Alert on low quality                │   │
│   │   • Gap detection                                                              │   │
│   │                                                                                 │   │
│   └─────────────────────────────────┬───────────────────────────────────────────────┘   │
│                                     │                                                   │
│                                     ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         3. RECONCILIATION                                       │   │
│   │                    (data/pipeline/reconciliation.py)                            │   │
│   │                                                                                 │   │
│   │   If multiple sources:                    Confidence Scores:                    │   │
│   │   • Compare prices at same timestamps     • Databento: 1.0                      │   │
│   │   • Detect discrepancies (>0.001%)        • Binance: 1.0                        │   │
│   │   • Use higher confidence source          • Polygon: 0.95                       │   │
│   │   • Fill gaps from secondary              • Yahoo: 0.75                         │   │
│   │   • Log to audit trail                                                         │   │
│   │   • Alert if discrepancy > 0.1%                                                │   │
│   │                                                                                 │   │
│   └─────────────────────────────────┬───────────────────────────────────────────────┘   │
│                                     │                                                   │
│                                     ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                            4. STORAGE                                           │   │
│   │                      (data/pipeline/storage.py)                                 │   │
│   │                                                                                 │   │
│   │   ┌─────────────────────┐      ┌─────────────────────┐                         │   │
│   │   │    TIMESCALEDB      │      │      PARQUET        │                         │   │
│   │   │                     │      │                     │                         │   │
│   │   │  raw_ohlcv_*        │      │  /app/data/catalog/ │                         │   │
│   │   │  (30 days)          │      │                     │                         │   │
│   │   │                     │      │  SPY/               │                         │   │
│   │   │  clean_ohlcv_*      │      │    SPY_1h.parquet   │                         │   │
│   │   │  (indefinite)       │      │    SPY_1d.parquet   │                         │   │
│   │   │                     │      │  BTCUSDT/           │                         │   │
│   │   │  data_pipeline_     │      │    BTCUSDT_1h.pq    │                         │   │
│   │   │  metadata           │      │                     │                         │   │
│   │   └─────────┬───────────┘      └──────────┬──────────┘                         │   │
│   │             │                              │                                    │   │
│   └─────────────┼──────────────────────────────┼────────────────────────────────────┘   │
│                 │                              │                                        │
└─────────────────┼──────────────────────────────┼────────────────────────────────────────┘
                  │                              │
                  │         ┌────────────────────┘
                  │         │
                  ▼         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           NAUTILUS TRADER LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         BACKTEST ENGINE                                         │   │
│   │                    (config/backtest.py)                                         │   │
│   │                                                                                 │   │
│   │   ┌─────────────────────┐      ┌─────────────────────┐                         │   │
│   │   │  ParquetDataCatalog │      │  BacktestNode       │                         │   │
│   │   │                     │ ───► │                     │                         │   │
│   │   │  Reads from:        │      │  • Simulated fills  │                         │   │
│   │   │  /app/data/catalog  │      │  • Order matching   │                         │   │
│   │   │                     │      │  • PnL tracking     │                         │   │
│   │   └─────────────────────┘      └──────────┬──────────┘                         │   │
│   │                                           │                                     │   │
│   └───────────────────────────────────────────┼─────────────────────────────────────┘   │
│                                               │                                         │
│   ┌───────────────────────────────────────────┼─────────────────────────────────────┐   │
│   │                          LIVE ENGINE      │                                     │   │
│   │                      (config/live.py)     │                                     │   │
│   │                                           │                                     │   │
│   │   ┌─────────────────────┐      ┌──────────▼──────────┐                         │   │
│   │   │  IBKRDataClient     │      │    TradingNode      │                         │   │
│   │   │  (real-time bars)   │ ───► │                     │                         │   │
│   │   │                     │      │  • Live execution   │                         │   │
│   │   │  IBKRExecClient     │ ◄─── │  • Position mgmt    │                         │   │
│   │   │  (order execution)  │      │  • Risk controls    │                         │   │
│   │   └─────────────────────┘      └──────────┬──────────┘                         │   │
│   │                                           │                                     │   │
│   │   ┌─────────────────────┐                 │                                     │   │
│   │   │  BinanceDataClient  │                 │ (for crypto live trading)          │   │
│   │   │  BinanceExecClient  │ ◄───────────────┘                                    │   │
│   │   └─────────────────────┘                                                      │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────┬───────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                             GYMNASIUM ENVIRONMENT                                        │
│                            (gym_env/nautilus_env.py)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         NautilusGymEnv                                          │   │
│   │                                                                                 │   │
│   │   Wraps NautilusTrader backtest engine as Gymnasium environment                │   │
│   │                                                                                 │   │
│   │   ┌───────────────────────────────────────────────────────────────────────┐    │   │
│   │   │  OBSERVATION SPACE (45 features)                                      │    │   │
│   │   │                                                                       │    │   │
│   │   │  Price Features:        Technical:           Market State:            │    │   │
│   │   │  • Returns (1,5,10,20)  • RSI (14)          • Position size          │    │   │
│   │   │  • Log returns          • MACD              • Unrealized PnL         │    │   │
│   │   │  • Volatility           • Bollinger %B      • Time to close          │    │   │
│   │   │  • Price vs SMA         • ATR               • Day of week            │    │   │
│   │   │                                                                       │    │   │
│   │   └───────────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                                 │   │
│   │   ┌───────────────────────────────────────────────────────────────────────┐    │   │
│   │   │  ACTION SPACE                                                         │    │   │
│   │   │                                                                       │    │   │
│   │   │  Discrete(3): [SELL, HOLD, BUY]                                       │    │   │
│   │   │      or                                                               │    │   │
│   │   │  Box(-1, 1): Continuous position sizing                               │    │   │
│   │   │                                                                       │    │   │
│   │   └───────────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                                 │   │
│   │   ┌───────────────────────────────────────────────────────────────────────┐    │   │
│   │   │  REWARD FUNCTION                                                      │    │   │
│   │   │                                                                       │    │   │
│   │   │  reward = α * returns + β * sharpe_contribution - γ * drawdown        │    │   │
│   │   │                                                                       │    │   │
│   │   └───────────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────┬───────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              RL TRAINING (RunPod GPU)                                   │
│                            (training/train_batch.py)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                 │   │
│   │   500 AGENTS CONFIGURATION (config/agents_500_pro.yaml)                        │   │
│   │                                                                                 │   │
│   │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │   │
│   │   │  Agent 1    │ │  Agent 2    │ │  Agent 3    │ │  Agent N    │              │   │
│   │   │  PPO        │ │  SAC        │ │  A2C        │ │  ...        │              │   │
│   │   │  SPY        │ │  QQQ        │ │  BTCUSDT    │ │             │              │   │
│   │   │  1h         │ │  4h         │ │  1d         │ │             │              │   │
│   │   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘              │   │
│   │          │               │               │               │                      │   │
│   │          └───────────────┴───────────────┴───────────────┘                      │   │
│   │                                  │                                              │   │
│   │                                  ▼                                              │   │
│   │                    ┌─────────────────────────┐                                  │   │
│   │                    │  Stable-Baselines3      │                                  │   │
│   │                    │  • 5M steps per agent   │                                  │   │
│   │                    │  • MLflow tracking      │                                  │   │
│   │                    │  • Checkpointing        │                                  │   │
│   │                    └────────────┬────────────┘                                  │   │
│   │                                 │                                               │   │
│   │                                 ▼                                               │   │
│   │                    ┌─────────────────────────┐                                  │   │
│   │                    │   TRAINED MODELS        │                                  │   │
│   │                    │   /app/models/*.zip     │                                  │   │
│   │                    └────────────┬────────────┘                                  │   │
│   │                                 │                                               │   │
│   └─────────────────────────────────┼───────────────────────────────────────────────┘   │
│                                     │                                                   │
└─────────────────────────────────────┼───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              VALIDATION PIPELINE                                        │
│                          (validation/run_validation.py)                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         5 FILTROS DE VALIDACIÓN                                 │   │
│   │                                                                                 │   │
│   │   500 Models                                                                    │   │
│   │       │                                                                         │   │
│   │       ▼                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │ FILTRO 1: Métricas Básicas                                              │  │   │
│   │   │ • Sharpe > 1.5                                                          │  │   │
│   │   │ • Max Drawdown < 15%                                                    │  │   │
│   │   │ • Win Rate > 50%                                                        │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │       │ ~200 pass                                                               │   │
│   │       ▼                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │ FILTRO 2: Cross-Validation                                              │  │   │
│   │   │ • Test on unseen instruments                                            │  │   │
│   │   │ • Generalization check                                                  │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │       │ ~120 pass                                                               │   │
│   │       ▼                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │ FILTRO 3: Diversificación                                               │  │   │
│   │   │ • Correlation < 0.5 between agents                                      │  │   │
│   │   │ • Remove redundant strategies                                           │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │       │ ~70 pass                                                                │   │
│   │       ▼                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │ FILTRO 4: Walk-Forward                                                  │  │   │
│   │   │ • Train: 2015-2022                                                      │  │   │
│   │   │ • Validate: 2023                                                        │  │   │
│   │   │ • Test: 2024                                                            │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │       │ ~50 pass                                                                │   │
│   │       ▼                                                                         │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │ FILTRO 5: Paper Trading                                                 │  │   │
│   │   │ • 2-4 weeks live paper                                                  │  │   │
│   │   │ • Verify real-world behavior                                            │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │       │ ~30-50 VALIDATED MODELS                                                 │   │
│   │       ▼                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              LIVE TRADING SYSTEM                                        │
│                            (live/ + strategies/)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          MODEL LOADER                                           │   │
│   │                       (live/model_loader.py)                                    │   │
│   │                                                                                 │   │
│   │   Load 30-50 validated .zip models                                             │   │
│   │   Initialize with frozen weights (predict only)                                │   │
│   │                                                                                 │   │
│   └─────────────────────────────────┬───────────────────────────────────────────────┘   │
│                                     │                                                   │
│                                     ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          VOTING SYSTEM                                          │   │
│   │                       (live/voting_system.py)                                   │   │
│   │                                                                                 │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                              │   │
│   │   │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │Agent N  │                              │   │
│   │   │  BUY    │ │  BUY    │ │  HOLD   │ │  BUY    │                              │   │
│   │   │ conf:0.8│ │ conf:0.7│ │ conf:0.5│ │ conf:0.9│                              │   │
│   │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                              │   │
│   │        │           │           │           │                                    │   │
│   │        └───────────┴───────────┴───────────┘                                    │   │
│   │                         │                                                       │   │
│   │                         ▼                                                       │   │
│   │               ┌─────────────────────┐                                          │   │
│   │               │  Weighted Average   │                                          │   │
│   │               │  + Confidence Filter│                                          │   │
│   │               │  (threshold > 0.6)  │                                          │   │
│   │               └──────────┬──────────┘                                          │   │
│   │                          │                                                      │   │
│   │                          ▼                                                      │   │
│   │               ┌─────────────────────┐                                          │   │
│   │               │  FINAL SIGNAL: BUY  │                                          │   │
│   │               │  Confidence: 0.72   │                                          │   │
│   │               └──────────┬──────────┘                                          │   │
│   │                          │                                                      │   │
│   └──────────────────────────┼──────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         RISK MANAGER                                            │   │
│   │                      (live/risk_manager.py)                                     │   │
│   │                                                                                 │   │
│   │   Limits:                          Circuit Breakers:                           │   │
│   │   • Max position: $10,000          • 5 consecutive losses → pause 30min       │   │
│   │   • Max exposure: $50,000          • Daily loss > $500 → stop trading         │   │
│   │   • Max drawdown: 10%              • Drawdown > 10% → close all               │   │
│   │   • Risk per trade: 2%                                                         │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────┬───────────────────────────────────────────┘   │
│                                         │                                               │
│                                         ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                       RL TRADING STRATEGY                                       │   │
│   │                     (strategies/rl_strategy.py)                                 │   │
│   │                                                                                 │   │
│   │   Extends NautilusTrader Strategy class                                        │   │
│   │                                                                                 │   │
│   │   on_bar(bar):                                                                 │   │
│   │       features = extract_features(bar)                                         │   │
│   │       signal = voting_system.get_signal(features)                              │   │
│   │       if risk_manager.check_order(signal):                                     │   │
│   │           self.submit_order(signal)                                            │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────┬───────────────────────────────────────────┘   │
│                                         │                                               │
│                                         ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                        NAUTILUS TRADING NODE                                    │   │
│   │                          (main.py --live)                                       │   │
│   │                                                                                 │   │
│   │   TradingNode:                                                                 │   │
│   │   ├── DataClient: IBKR (real-time bars)                                        │   │
│   │   ├── ExecClient: IBKR (order execution)                                       │   │
│   │   ├── Strategy: RLTradingStrategy                                              │   │
│   │   └── RiskEngine: Built-in + Custom                                            │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────┬───────────────────────────────────────────┘   │
│                                         │                                               │
│                                         ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          BROKERS                                                │   │
│   │                                                                                 │   │
│   │   ┌─────────────────────┐      ┌─────────────────────┐                         │   │
│   │   │  INTERACTIVE BROKERS│      │      BINANCE        │                         │   │
│   │   │                     │      │                     │                         │   │
│   │   │  • Stocks           │      │  • Crypto Spot      │                         │   │
│   │   │  • Futures          │      │  • Crypto Futures   │                         │   │
│   │   │  • Options          │      │                     │                         │   │
│   │   │  • Forex            │      │                     │                         │   │
│   │   │                     │      │                     │                         │   │
│   │   │  Account: DU0275624│      │  Testnet → Live     │                         │   │
│   │   │  Port: 7497         │      │                     │                         │   │
│   │   └─────────────────────┘      └─────────────────────┘                         │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               MONITORING                                                │
│                         (monitoring/ + live/)                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                      │
│   │    GRAFANA      │   │    TELEGRAM     │   │     LOGS        │                      │
│   │                 │   │      BOT        │   │                 │                      │
│   │  • Performance  │   │                 │   │  /app/logs/     │                      │
│   │  • Positions    │   │  • Trade alerts │   │  • trading.log  │                      │
│   │  • Risk metrics │   │  • Error alerts │   │  • errors.log   │                      │
│   │  • Data health  │   │  • Daily P&L    │   │  • audit.jsonl  │                      │
│   │                 │   │  • Source down  │   │                 │                      │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘                      │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Connection Points Verification

### ✅ Data Sources → Pipeline
```
Sources (Binance, Yahoo, etc.)
    ↓ HTTP/REST APIs
data/adapters/*_adapter.py
    ↓ FetchResult (pd.DataFrame)
data/pipeline/manager.py
```

### ✅ Pipeline → Storage
```
data/pipeline/manager.py
    ↓ save_raw(), save_clean()
data/pipeline/storage.py
    ↓ SQL INSERT / Parquet write
TimescaleDB (clean_ohlcv_*) + /app/data/catalog/*.parquet
```

### ✅ Storage → NautilusTrader Backtest
```
/app/data/catalog/*.parquet
    ↓ ParquetDataCatalog.read()
nautilus_trader.persistence.catalog
    ↓ Bar objects
BacktestNode.run()
```

### ✅ NautilusTrader → Gymnasium
```
BacktestNode
    ↓ Wrapped as environment
gym_env/nautilus_env.py (NautilusGymEnv)
    ↓ Gymnasium interface (reset, step, render)
Stable-Baselines3 training
```

### ✅ Training → Validation → Live
```
training/train_batch.py
    ↓ .zip model files
/app/models/
    ↓ Load validated models
live/model_loader.py
    ↓ predict() calls
live/voting_system.py
    ↓ Aggregated signal
strategies/rl_strategy.py
    ↓ submit_order()
TradingNode (IBKR/Binance)
```

### ✅ Live Data Flow
```
IBKR Gateway (TWS 10.43.1c, port 7497)
    ↓ IBKRDataClient subscription
TradingNode.on_bar()
    ↓ Strategy callback
RLTradingStrategy.on_bar()
    ↓ Feature extraction + prediction
VotingSystem.get_signal()
    ↓ Risk check
RiskManager.check_order()
    ↓ Order submission
IBKRExecClient.submit_order()
```

## File Mapping

| Component | File(s) | Status |
|-----------|---------|--------|
| Data Adapters | `data/adapters/*.py` | ✅ Created |
| Pipeline Core | `data/pipeline/*.py` | ✅ Created |
| DB Schema | `infra/pipeline_schema.sql` | ✅ Created |
| Backtest Config | `config/backtest.py` | ✅ Exists |
| Live Config | `config/live.py` | ✅ Exists |
| Gym Environment | `gym_env/nautilus_env.py` | ✅ Verified (NautilusTrader native) |
| Training | `training/train_batch.py` | 📋 To verify |
| Validation | `validation/run_validation.py` | 📋 To verify |
| RL Strategy | `strategies/rl_strategy.py` | 📋 To verify |
| Voting System | `live/voting_system.py` | 📋 To verify |
| Risk Manager | `live/risk_manager.py` | ✅ Exists |
| Main Entry | `main.py` | ✅ Exists |
