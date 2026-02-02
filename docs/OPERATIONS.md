# NautilusTrader RL Agents - Operations Guide

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Training Agents](#training-agents)
5. [Validation Pipeline](#validation-pipeline)
6. [Live Trading](#live-trading)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Components

1. **Training (RunPod GPU)**
   - FinRL / Stable-Baselines3
   - 500 agents training
   - Gymnasium environment over NautilusTrader

2. **Live Trading (VPS Hetzner)**
   - NautilusTrader execution
   - Voting system
   - Risk management
   - IBKR + Binance adapters

3. **Data Pipeline**
   - TimescaleDB storage
   - Polygon, EOD, Binance providers
   - Parquet catalog for backtests

4. **Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Telegram alerts

### Data Flow

```
Market Data → TimescaleDB → Parquet Catalog → Training
                   ↓
            NautilusTrader ← Validated Models
                   ↓
              Brokers (IBKR/Binance)
```

---

## Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- 8GB+ RAM
- IBKR Gateway (for IBKR trading)

### VPS Setup

```bash
# Clone or create directory
mkdir -p /opt/nautilus-agents
cd /opt/nautilus-agents

# Copy all project files
# (from local or git clone)

# Create .env from example
cp .env.example .env
nano .env  # Edit with your credentials

# Build and start
docker-compose build
docker-compose up -d
```

### Environment Variables

```bash
# Required
IBKR_ACCOUNT=UXXXXXX
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=bot_token
TELEGRAM_CHAT_ID=chat_id

# Optional
TRADING_MODE=paper  # paper or live
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=500
```

---

## Configuration

### Trading Node (`config/live.py`)

Key settings:
- `IBKR_PORT`: 7497 for paper, 7496 for live
- `BINANCE_TESTNET`: true for testnet
- `max_position_size`: Maximum per position
- `risk_per_trade`: Risk per trade (fraction)

### Instruments (`config/instruments.py`)

Add new instruments:
```python
InstrumentConfig(
    symbol="NEW_SYMBOL",
    venue="IBKR",
    asset_class="EQUITY",
    price_precision=2,
    tick_size=0.01,
)
```

### Risk Limits (`live/risk_manager.py`)

```python
RiskLimits(
    max_daily_loss=500.0,
    max_drawdown=0.10,
    max_positions=10,
    consecutive_losses_limit=5,
)
```

---

## Training Agents

### Local Training

```bash
# Single agent
python training/train_agent.py \
  --agent-id agent_SPY_test \
  --symbol SPY \
  --timesteps 1000000

# Batch training
python training/train_batch.py \
  --config config/agents_500_pro.yaml \
  --parallel 4
```

### RunPod Training

```bash
# Estimate cost
python training/runpod_launcher.py --estimate --agents 0-99

# Launch training
python training/runpod_launcher.py --agents 0-99 --gpu-type A100

# Distributed (500 agents across pods)
python training/runpod_launcher.py --distributed 500 --agents-per-pod 50
```

### Download Models

```bash
# From cloud storage
python training/download_models.py --download

# From RunPod
rsync -avz runpod:/workspace/models/ ./models/
```

---

## Validation Pipeline

### Full Pipeline

```bash
python validation/run_validation.py

# Expected flow:
# 500 agents → Filter 1 (Basic) → ~200
#           → Filter 2 (Cross-Val) → ~150
#           → Filter 3 (Diversity) → 50
#           → Filter 4 (Walk-Forward) → ~40
#           → Filter 5 (Paper Trading) → ~30-50
```

### Individual Filters

```bash
# Basic metrics only
python validation/run_validation.py --filter basic

# Cross-validation
python validation/run_validation.py --filter crossval

# Diversity selection
python validation/run_validation.py --filter diversity --target-agents 50

# Walk-forward testing
python validation/run_validation.py --filter walkforward

# Paper trading (start session)
python validation/run_validation.py --filter paper

# Paper trading (validate after 2-4 weeks)
python validation/run_validation.py --filter paper --paper-session SESSION_ID
```

### Validation Criteria

| Filter | Criteria |
|--------|----------|
| Basic | Sharpe > 1.5, DD < 15%, WR > 50% |
| Cross-Val | Performance on unseen instruments |
| Diversity | Correlation < 0.5 between agents |
| Walk-Forward | 2024 out-of-sample performance |
| Paper Trading | 2-4 weeks real-time validation |

---

## Live Trading

### Starting Live Trading

```bash
# Paper trading mode
docker-compose up -d

# Live trading mode
# Edit .env: TRADING_MODE=live
docker-compose down
docker-compose up -d
```

### Monitoring Live Trading

```bash
# Real-time logs
docker logs -f nautilus-agents

# Health check
curl http://localhost:8000/health | jq

# Metrics
curl http://localhost:8000/metrics
```

### Emergency Stop

```bash
# Graceful stop
docker-compose stop

# Force stop
docker-compose down

# Close all positions (from container)
docker exec nautilus-agents python -c "
from live.risk_manager import RiskManager
rm = RiskManager()
rm.close_all_positions()
"
```

---

## Monitoring

### Grafana Setup

1. Access: `http://[VPS_IP]:3000`
2. Default login: admin/admin
3. Import dashboard from `monitoring/grafana/dashboards/trading_dashboard.json`

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `nautilus_agents_equity` | Current equity | - |
| `nautilus_agents_daily_pnl` | Today's PnL | < -$300 |
| `nautilus_agents_drawdown` | Current DD | > 8% |
| `nautilus_agents_active_models` | Loaded models | < 10 |

### Telegram Alerts

Configured in `.env`:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Alerts sent for:
- Trade executions
- Circuit breaker activations
- System health issues
- Daily summary (9 AM UTC)

---

## Troubleshooting

### IBKR Connection Issues

**Problem:** Cannot connect to IBKR Gateway

**Solutions:**
1. Verify Gateway is running on correct port (7497/7496)
2. Check `IBKR_HOST` in `.env`
3. Ensure API connections enabled in Gateway settings
4. Check firewall rules

```bash
# Test connection
nc -zv localhost 7497
```

### Model Loading Failures

**Problem:** Models not loading or predicting

**Solutions:**
1. Check model files exist in `/opt/nautilus-agents/models/`
2. Verify model format (SB3 .zip files)
3. Check validation file

```bash
# List models
ls -la /opt/nautilus-agents/models/

# Test model loading
docker exec nautilus-agents python -c "
from stable_baselines3 import PPO
model = PPO.load('models/agent_SPY_1h_001/agent_SPY_1h_001_final.zip')
print('Model loaded successfully')
"
```

### High Memory Usage

**Problem:** Container using too much memory

**Solutions:**
1. Reduce number of active models
2. Increase swap space
3. Adjust container memory limit

```bash
# Check memory
docker stats nautilus-agents

# Adjust in docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 4G
```

### Database Connection Issues

**Problem:** Cannot connect to TimescaleDB

**Solutions:**
1. Verify database is running
2. Check connection string in `.env`
3. Test connection directly

```bash
# Test database
docker exec nautilus-agents python -c "
from data.adapters.timescale_adapter import TimescaleAdapter
adapter = TimescaleAdapter()
print(adapter.get_available_symbols())
adapter.close()
"
```

### Voting System Issues

**Problem:** No signals being generated

**Solutions:**
1. Check model predictions are working
2. Verify confidence threshold (`min_confidence=0.6`)
3. Check voting method configuration

```bash
# Debug voting
docker exec nautilus-agents python -c "
from live.voting_system import VotingSystem
from live.model_loader import load_validated_models
import numpy as np

models = load_validated_models(instrument_id='SPY.IBKR')
vs = VotingSystem(models)
obs = np.random.randn(45).astype(np.float32)
result = vs.vote(obs)
print(f'Signal: {result.signal}, Confidence: {result.confidence}')
"
```

---

## Maintenance Tasks

### Weekly

- [ ] Review Grafana dashboards
- [ ] Check model performance
- [ ] Verify data sync is working
- [ ] Review Telegram alert history

### Monthly

- [ ] Run validation pipeline on all agents
- [ ] Update models if needed
- [ ] Review and adjust risk parameters
- [ ] Check system resource trends

### Quarterly

- [ ] Retrain agents with new data
- [ ] Full system backup
- [ ] Review and update documentation
- [ ] Security audit

---

## Support

For issues not covered in this guide:

1. Check logs: `docker logs nautilus-agents`
2. Review RUNBOOK.md for incident response
3. Check NautilusTrader documentation: https://nautilustrader.io
4. Contact system administrator
