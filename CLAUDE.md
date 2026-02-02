# Nautilus-Agents Project Context

> **SISTEMA AUTONOMO CON GOVERNANCE**
> - Operacion autonoma sin aprobacion humana
> - Validacion automatica entre roles
> - Reglas inmutables en `.rules/IMMUTABLE_RULES.md`
> - Solo alertas CRITICAS van a Telegram
>
> **Decisiones internas vs externas:**
> - Las decisiones internas (documentar, loguear, actualizar estado) se ejecutan automáticamente
> - Solo consultar al usuario para recursos externos (API keys, capital, infraestructura)

## [CRITICAL] MANDATORY: Session Start Protocol

**ANTES DE CUALQUIER ACCION**, ejecutar:

```bash
python scripts/session_start.py
```

Este script es OBLIGATORIO. Realiza:
1. Verificacion de integridad de archivos protegidos
2. Verificacion de IMMUTABLE_RULES.md
3. Inicializacion del motor de governance
4. Carga del estado actual de roles
5. Marcado de sesion activa

**Si no se ejecuta**: El sistema registra "Session without protocol" en WARNINGS.md

## Quick Start for New Session

```
1. python scripts/session_start.py     <-- OBLIGATORIO
2. Read .rules/IMMUTABLE_RULES.md
3. Load config: config/autonomous_config.yaml
4. Check role's STATE.md for current work
5. EXECUTE through governance pipeline
6. Document in DECISIONS.md or LESSONS_LEARNED.md
7. Alert ONLY if critical
```

## Protected Files

Los siguientes archivos estan protegidos:
- `.rules/IMMUTABLE_RULES.md` - INMUTABLE (no puede modificarse)
- `CLAUDE.md` - Protegido (requiere proceso formal)
- `config/autonomous_config.yaml` - Protegido
- `governance/governance_engine.py` - Protegido

Los checksums se verifican en cada sesion. Ver: `.rules/CHECKSUMS.md`

## Governance Flow

```
Cambio propuesto
      │
      ▼
┌─────────────────┐
│ governance/     │
│ ci_cd.py        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Role Validators                      │
│ • rl_validator.py                   │
│ • quant_validator.py                │
│ • mlops_validator.py                │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  PASS      FAIL
    │         │
    ▼         ▼
 Auto-     Auto-
 Merge     Revert
    │         │
    ▼         ▼
DECISIONS  LESSONS_
.md        LEARNED.md
```

## Dynamic Role System

This project uses a **collaborative role system** in `.roles/`:

```
.roles/
├── TEAM_PROTOCOL.md      # How roles collaborate
├── DECISION_QUEUE.md     # Pending cross-role decisions
├── DECISIONS.md          # Historical decisions log
├── LESSONS_LEARNED.md    # Continuous improvement
├── rl_engineer/          # RL Engineer state & expertise
├── quant_developer/      # Quant Developer state & expertise
├── mlops_engineer/       # MLOps Engineer state & expertise
└── retrospectives/       # Sprint reviews
```

### Invoke a Role
- `"@rl_engineer continúa con validation"` → Work as RL Engineer
- `"@quant_developer implementa voting system"` → Work as Quant Dev
- `"@team_review decisión sobre thresholds"` → Multi-role discussion

## Project Overview
Sistema de 500 agentes RL para trading algorítmico usando NautilusTrader como motor de ejecución.

## Active Roles

### RL Engineer (Primary)
- **Scope**: Gymnasium environment, reward functions, training pipeline, agent configuration
- **Key Files**: `gym_env/`, `training/`, `configs/agents_500_pro.yaml`
- **Current Focus**: Training pipeline functional, next is validation filters

### Quant Developer (Primary)
- **Scope**: NautilusTrader integration, backtesting, strategies, order execution
- **Key Files**: `strategies/`, `data/`, `live/`
- **Current Focus**: BacktestEngine working, next is live strategy

### MLOps Engineer (Secondary)
- **Scope**: Infrastructure, Docker, RunPod, MLflow, monitoring
- **Key Files**: `docker-compose.yml`, `monitoring/`, deployment scripts
- **Current Focus**: Local setup complete, next is RunPod deployment

---

## Project Status

### Completed Phases
- [x] Phase 1: NautilusTrader Setup
- [x] Phase 2: Data Integration (ParquetDataCatalog)
- [x] Phase 3: Gymnasium Environment
- [x] Phase 4: Training Pipeline (basic)

### In Progress
- [ ] Phase 5: Validation 5 Filters
- [ ] Phase 6: Voting System + Live Strategy
- [ ] Phase 7: Monitoring + Production

### Key Decisions Made
| Decision | Value | Date |
|----------|-------|------|
| Execution Engine | NautilusTrader 1.221.0 | 2026-02 |
| RL Framework | Stable-Baselines3 | 2026-02 |
| Data Format | ParquetDataCatalog | 2026-02 |
| Markets | IBKR + Binance | 2026-02 |
| Deployment | Docker (local) + RunPod (GPU) | 2026-02 |

---

## Architecture

```
nautilus-agents/
├── gym_env/          # Gymnasium environment (RL Engineer)
│   ├── nautilus_env.py    # Main env wrapping BacktestEngine
│   ├── observation.py     # Feature extraction (45 features)
│   └── rewards.py         # Reward functions (sharpe, sortino, pnl)
│
├── training/         # Training pipeline (RL Engineer + MLOps)
│   ├── train_agent.py     # Single agent training
│   ├── train_batch.py     # Batch training
│   └── runpod_launcher.py # GPU deployment
│
├── strategies/       # Trading strategies (Quant Developer)
│   └── rl_strategy.py     # RLTradingStrategy for live
│
├── validation/       # 5-filter validation (RL Engineer)
│   ├── filter_1_basic.py      # Sharpe > 1.5, DD < 15%
│   ├── filter_2_cross_val.py  # Cross-market validation
│   ├── filter_3_diversity.py  # Correlation < 0.5
│   ├── filter_4_walkforward.py
│   └── filter_5_paper.py
│
├── live/            # Live trading (Quant Developer + MLOps)
│   ├── voting_system.py   # Signal aggregation
│   └── risk_manager.py    # Position limits
│
├── data/            # Data pipeline
│   └── catalog/     # ParquetDataCatalog
│
└── configs/
    └── agents_500_pro.yaml  # Agent configurations
```

---

## API Compatibility Notes (NautilusTrader 1.221.0)

### MarginAccount
- NO `equity()` method - use `balance_total(currency).as_double()`
- `balance(currency)` returns `AccountBalance` object
- Use `AccountBalance.free.as_double()` for free cash

### BacktestVenueConfig
- NO `maker_fee`/`taker_fee` - fees on instrument level
- `base_currency` should be `None` (uses instrument's currency)
- `default_leverage` must be `Decimal`, not `float`

### StrategyConfig
- NO custom params in constructor
- Create separate dataclass for strategy params

---

## Next Actions by Role

### RL Engineer
1. Implement `validation/filter_1_basic.py`
2. Run validation on trained models
3. Document validation thresholds

### Quant Developer
1. Complete `strategies/rl_strategy.py` for live trading
2. Implement `live/voting_system.py`
3. Test with paper trading

### MLOps Engineer
1. Create RunPod training scripts
2. Setup MLflow tracking server
3. Configure Grafana dashboards

---

## Commands Reference

```bash
# Populate data catalog
python scripts/populate_catalog_standalone.py --preset crypto --start 2024-01-01

# Test data flow
python scripts/test_data_flow.py

# Train single agent
python training/train_agent.py --agent-id agent_001 --symbol SPY

# Train batch (not yet implemented fully)
python training/train_batch.py --config configs/agents_500_pro.yaml
```

---

## Contact Points
- Plan file: `~/.claude/plans/misty-cooking-waterfall.md`
- Config: `configs/agents_500_pro.yaml`
- Tests: `scripts/test_data_flow.py`
