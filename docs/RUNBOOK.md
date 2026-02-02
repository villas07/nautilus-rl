# NautilusTrader RL Agents - Runbook

## System Overview

This system runs 500 RL agents for automated trading across IBKR and Binance markets.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  VPS (Hetzner)                                              │
│                                                             │
│  ┌─────────────────┐   ┌──────────────────────────────────┐ │
│  │  Data Pipeline  │   │  NautilusTrader Container        │ │
│  │  - TimescaleDB  │──▶│  - 50 Validated Agents           │ │
│  │  - Polygon      │   │  - Voting System                 │ │
│  │  - Binance      │   │  - Risk Management               │ │
│  └─────────────────┘   └──────────┬───────────────────────┘ │
│                                   │                         │
│  ┌─────────────────┐              │                         │
│  │  Monitoring     │◀─────────────┘                         │
│  │  - Grafana      │                                        │
│  │  - Telegram     │                                        │
│  └─────────────────┘                                        │
└───────────────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  IBKR Gateway   │
                          │  Binance API    │
                          └─────────────────┘
```

## Quick Reference

### Locations

| Component | Path |
|-----------|------|
| Docker compose | `/opt/nautilus-agents/docker-compose.yml` |
| Models | `/opt/nautilus-agents/models/` |
| Logs | `/opt/nautilus-agents/logs/` |
| Data catalog | `/opt/nautilus-agents/data/catalog/` |
| Config | `/opt/nautilus-agents/config/` |

### Ports

| Service | Port |
|---------|------|
| Health/Metrics | 8000 |
| IBKR Gateway | 7497 |
| Grafana | 3000 |
| TimescaleDB | 5432 |

### Key Files

| File | Purpose |
|------|---------|
| `.env` | Environment configuration |
| `config/live.py` | Trading node configuration |
| `config/agents_500_pro.yaml` | Agent definitions |
| `validation/latest.json` | Validated agents list |

---

## Daily Operations

### Morning Checklist

1. **Check system health**
   ```bash
   curl http://localhost:8000/health
   docker logs nautilus-agents --tail 100
   ```

2. **Verify connections**
   - IBKR Gateway connected
   - Binance API responding
   - Database accessible

3. **Check overnight trades**
   - Review Telegram alerts
   - Check Grafana dashboard
   - Verify positions match expected

### Status Commands

```bash
# Container status
docker ps | grep nautilus

# View logs
docker logs -f nautilus-agents

# Health check
curl http://localhost:8000/health | jq

# Metrics
curl http://localhost:8000/metrics

# Database status
docker exec nautilus-agents python -c "from data.adapters import TimescaleAdapter; print(TimescaleAdapter().get_available_symbols())"
```

---

## Incident Response

### 1. Trading Not Executing

**Symptoms:**
- No trades in logs
- Positions not changing
- Grafana shows no activity

**Diagnosis:**
```bash
# Check container
docker logs nautilus-agents --tail 500 | grep -i error

# Check IBKR connection
docker exec nautilus-agents python -c "from config.live import get_ibkr_data_config; print('ok')"

# Check circuit breaker
curl http://localhost:8000/health | jq '.checks.risk_manager'
```

**Resolution:**
1. If IBKR disconnected: Restart IB Gateway
2. If circuit breaker active: Wait for cooldown or manual reset
3. If model issue: Check model loading logs

### 2. High Latency

**Symptoms:**
- Slow order execution
- Prediction latency > 100ms
- Grafana shows latency spikes

**Diagnosis:**
```bash
# Check system resources
docker stats nautilus-agents

# Check prediction latency
curl http://localhost:8000/metrics | grep prediction_latency
```

**Resolution:**
1. If CPU > 80%: Reduce active models
2. If memory > 90%: Restart container
3. If network: Check IBKR/Binance connectivity

### 3. Excessive Losses

**Symptoms:**
- Daily PnL significantly negative
- Drawdown > 10%
- Circuit breaker triggered

**Immediate Actions:**
```bash
# Emergency stop
docker exec nautilus-agents python -c "from live.risk_manager import RiskManager; rm = RiskManager(); rm.close_all_positions()"

# Or stop container
docker stop nautilus-agents
```

**Investigation:**
1. Review trades in logs
2. Check market conditions
3. Analyze model predictions
4. Verify data quality

### 4. Container Crashes

**Symptoms:**
- Container not running
- Automatic restart loops
- Memory errors in logs

**Diagnosis:**
```bash
# Check container status
docker ps -a | grep nautilus

# Check exit reason
docker inspect nautilus-agents | jq '.[0].State'

# Check logs before crash
docker logs nautilus-agents --tail 1000 | tail -200
```

**Resolution:**
1. Check available memory: `free -h`
2. Check disk space: `df -h`
3. If OOM: Increase container memory limit
4. If config issue: Verify `.env` and config files

---

## Restart Procedures

### Graceful Restart

```bash
cd /opt/nautilus-agents

# Stop gracefully (waits for positions to close)
docker-compose stop nautilus-trader

# Wait for confirmation
docker logs nautilus-agents --tail 50

# Start again
docker-compose up -d nautilus-trader
```

### Emergency Restart

```bash
# Force stop and restart
docker-compose restart nautilus-trader

# Or complete recreation
docker-compose down
docker-compose up -d
```

### Full System Restart

```bash
cd /opt/nautilus-agents

# Stop all services
docker-compose down

# Clear logs if needed
rm -rf logs/*.log

# Start fresh
docker-compose up -d

# Monitor startup
docker-compose logs -f
```

---

## Model Management

### Reload Models

```bash
# Hot reload (without restart)
docker exec nautilus-agents python -c "
from live.model_loader import ModelLoader
loader = ModelLoader()
loader.reload_all()
print('Models reloaded')
"
```

### Add New Validated Models

1. Copy models to `/opt/nautilus-agents/models/`
2. Update validation file or reload

```bash
# Copy from training server
rsync -avz runpod:/workspace/models/ /opt/nautilus-agents/models/

# Reload
docker exec nautilus-agents python -c "from live.model_loader import ModelLoader; ModelLoader().reload_all()"
```

### Check Model Status

```bash
docker exec nautilus-agents python -c "
from live.model_loader import ModelLoader
loader = ModelLoader()
stats = loader.get_model_stats()
print(f'Total validated: {stats[\"total_validated\"]}')
print(f'Currently cached: {stats[\"currently_cached\"]}')
print(f'By symbol: {stats[\"by_symbol\"]}')
"
```

---

## Monitoring

### Grafana Dashboard

Access: `http://[VPS_IP]:3000`

Key panels:
- **Equity Curve**: Overall performance
- **Daily P&L**: Today's performance
- **Drawdown**: Risk indicator
- **Trade Activity**: Trading frequency
- **Model Predictions**: Signal distribution

### Alerts

Telegram alerts for:
- Trade executions
- Circuit breaker triggers
- System health issues
- Daily summaries

### Log Analysis

```bash
# Trade logs
grep "Trade executed" /opt/nautilus-agents/logs/nautilus_live.log

# Error logs
grep -i error /opt/nautilus-agents/logs/nautilus_live.log | tail -50

# Signal logs
grep "Signal generated" /opt/nautilus-agents/logs/nautilus_live.log | tail -20
```

---

## Scheduled Tasks

### Cron Jobs

```cron
# Data sync (every 6 hours)
0 */6 * * * cd /opt/nautilus-agents && docker exec nautilus-agents python data/sync_data.py --incremental

# Daily health report (9 AM UTC)
0 9 * * * cd /opt/nautilus-agents && docker exec nautilus-agents python -c "from monitoring.health_checks import run_health_checks; import asyncio; asyncio.run(run_health_checks())"

# Weekly model check (Sunday 2 AM)
0 2 * * 0 cd /opt/nautilus-agents && docker exec nautilus-agents python validation/run_validation.py --filter basic
```

---

## Backup & Recovery

### Backup Locations

```bash
# Create backup
tar -czvf backup_$(date +%Y%m%d).tar.gz \
  /opt/nautilus-agents/models \
  /opt/nautilus-agents/config \
  /opt/nautilus-agents/.env
```

### Recovery from Backup

```bash
# Stop system
docker-compose down

# Restore from backup
tar -xzvf backup_YYYYMMDD.tar.gz -C /

# Restart
docker-compose up -d
```

---

## Contacts

| Role | Contact |
|------|---------|
| Primary On-Call | [Your contact] |
| IBKR Support | support@interactivebrokers.com |
| Binance Support | https://www.binance.com/en/support |
| Hetzner Support | support@hetzner.com |

---

## Appendix: Common Commands

```bash
# Quick status
docker ps && curl -s localhost:8000/health | jq

# View last trades
docker logs nautilus-agents 2>&1 | grep "Trade" | tail -20

# Check positions
curl -s localhost:8000/metrics | grep position_value

# Manual trade (emergency)
docker exec -it nautilus-agents python -c "
# Use IBKR API directly for manual interventions
"
```
