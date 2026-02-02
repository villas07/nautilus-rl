# Decisions Log

Historical record of all decisions made. Used for:
- Understanding why things are the way they are
- Learning from past decisions
- Onboarding new sessions with context

---

## 2026-02

### D-001: NautilusTrader as Execution Engine
- **Date**: 2026-02-01
- **Decided By**: @quant_developer, @rl_engineer
- **Choice**: NautilusTrader 1.221.0 over alternatives
- **Alternatives Considered**:
  - Backtrader (rejected: no native RL support)
  - VectorBT (rejected: limited live trading)
  - Custom engine (rejected: too much work)
- **Rationale**:
  - Native Rust performance
  - ParquetDataCatalog for efficient data
  - BacktestEngine integrates with Gymnasium
  - Live trading adapters (IBKR, Binance)
- **Impact**: All strategy code uses Nautilus APIs
- **Lessons**: Nautilus API changes frequently, document compatibility

### D-002: Stable-Baselines3 for RL
- **Date**: 2026-02-01
- **Decided By**: @rl_engineer
- **Choice**: SB3 over FinRL, RLlib, CleanRL
- **Rationale**:
  - Well-documented, stable API
  - Easy integration with Gymnasium
  - PPO, A2C, SAC all available
  - Good callback system for checkpointing
- **Impact**: Training code uses SB3 patterns

### D-003: Docker Deployment Strategy
- **Date**: 2026-02-01
- **Decided By**: @mlops_engineer
- **Choice**: Isolated Docker containers
- **Rationale**:
  - Reproducible environments
  - Easy deployment to RunPod
  - Separation from DeskGrade system
- **Impact**: All services containerized

### D-004: MarginAccount API Workaround
- **Date**: 2026-02-02
- **Decided By**: @rl_engineer
- **Choice**: Use `balance_total(currency).as_double()` instead of `equity()`
- **Rationale**:
  - NautilusTrader 1.221.0 changed API
  - `equity()` method no longer exists on MarginAccount
- **Impact**: All account balance queries updated
- **Files Changed**: `gym_env/nautilus_env.py:229-244`

### D-005: Bar Precision = 2
- **Date**: 2026-02-02
- **Decided By**: @quant_developer
- **Choice**: Hardcode price precision to 2 decimals
- **Rationale**:
  - Instrument price_precision is 2
  - Bar prices must match instrument precision
  - Avoids precision mismatch errors
- **Impact**: `scripts/populate_catalog_standalone.py:259`

### D-006: Autonomous Operation Mode
- **Date**: 2026-02-02
- **Decided By**: User directive
- **Choice**: Full autonomous operation with supervisor oversight
- **Rationale**:
  - User doesn't want to approve each decision
  - System should act on predefined rules
  - Only critical alerts go to Telegram
  - User supervises results via dashboards
- **Impact**: All DQ decisions auto-resolved via config
- **Config File**: `config/autonomous_config.yaml`

### D-007: Validation Thresholds (DQ-001)
- **Date**: 2026-02-02
- **Decided By**: Auto-configured
- **Choice**: Moderate thresholds
- **Values**:
  - Sharpe Ratio > 1.5
  - Max Drawdown < 15%
  - Win Rate > 50%
  - Profit Factor > 1.2
  - Min Trades: 50
- **Auto-action**: Failed agents archived (not deleted)

### D-008: Position Sizing (DQ-002)
- **Date**: 2026-02-02
- **Decided By**: Auto-configured
- **Choice**: Confidence-scaled positioning
- **Values**:
  - Base: $1,000 per signal
  - Max per symbol: $5,000
  - Max total exposure: $50,000
  - Min confidence: 60% (below = no trade)
  - Scaling: 50% at 70%, 100% at 80%, 150% at 90%
- **Auto-action**: Skip low-confidence trades

### D-009: Training Infrastructure (DQ-003)
- **Date**: 2026-02-02
- **Decided By**: Auto-configured
- **Choice**: Hybrid (local + RunPod)
- **Trigger for RunPod**: >10 agents OR >10M total timesteps
- **Safeguards**:
  - Max $50 per RunPod run
  - Auto-shutdown after 2 hours
- **Auto-action**: System switches infrastructure automatically

### D-010: Add EOD Historical Data Source
- **Date**: 2026-02-02
- **Decision Queue**: DQ-004
- **Evaluated By**: @quant_developer, @rl_engineer
- **Choice**: Add EOD Historical Data API for Europe and Asia markets
- **Rationale**:
  - Expands data coverage beyond US markets
  - Europe: LSE, XETRA, Euronext, SIX, BME, etc.
  - Asia: TSE, HKEX, SSE, SZSE, NSE, KRX, etc.
  - 70+ exchanges, 150,000+ tickers
  - Enables cross-market validation for RL agents
- **Cost**: ~$80/month
- **Implementation**: `data/adapters/eod_adapter.py`
- **Config**: EOD_API_KEY in .env
- **Status**: APPROVED

---


### CHG-20260202-090627: 5 files changed: code (4 files)
- **Date**: 2026-02-02 09:06
- **Proposed By**: @mlops_engineer
- **Validated By**: @mlops_engineer
- **Status**: AUTO-APPROVED
- **Files**: governance/ci_cd.py, governance/governance_engine.py, governance/hooks/install_hooks.py, governance/hooks/pre-commit, governance/validators/rl_validator.py
- **Validations**:
  - mlops_engineer: pass (2/2 checks)


### CHG-20260202-090711: 1 files changed: code (1 files)
- **Date**: 2026-02-02 09:07
- **Proposed By**: @rl_engineer
- **Validated By**: @rl_engineer
- **Status**: AUTO-APPROVED
- **Files**: validation/filter_1_basic.py
- **Validations**:
  - rl_engineer: pass (2/2 checks)


### CHG-20260202-091208: 15 files changed: code (13 files)
- **Date**: 2026-02-02 09:12
- **Proposed By**: @mlops_engineer
- **Validated By**: @mlops_engineer
- **Status**: AUTO-APPROVED
- **Files**: .roles/DECISIONS.md, .roles/LESSONS_LEARNED.md, .rules/.checksums.json, .rules/.session_active, .rules/CHECKSUMS.md...
- **Validations**:
  - mlops_engineer: pass (2/2 checks)


### CHG-20260202-092130: 8 files changed: code (5 files)
- **Date**: 2026-02-02 09:21
- **Proposed By**: @quant_developer
- **Validated By**: @quant_developer, @mlops_engineer
- **Status**: AUTO-APPROVED
- **Files**: .env.example, .roles/DECISIONS.md, .roles/DECISION_QUEUE.md, .roles/LESSONS_LEARNED.md, config/data_sources.py...
- **Validations**:
  - quant_developer: pass (2/2 checks)
  - mlops_engineer: pass (2/2 checks)


### CHG-20260202-093035: 1 files changed: code (1 files)
- **Date**: 2026-02-02 09:30
- **Proposed By**: @rl_engineer
- **Validated By**: @rl_engineer
- **Status**: AUTO-APPROVED
- **Files**: scripts/load_eod_to_catalog.py
- **Validations**:
  - rl_engineer: pass (2/2 checks)


### CHG-20260202-093317: 93 files changed: code (93 files)
- **Date**: 2026-02-02 09:33
- **Proposed By**: @quant_developer
- **Validated By**: @quant_developer
- **Status**: AUTO-APPROVED
- **Files**: data/eod/0005_HK_1d.csv, data/eod/000660_KO_1d.csv, data/eod/003550_KO_1d.csv, data/eod/005380_KO_1d.csv, data/eod/005930_KO_1d.csv...
- **Validations**:
  - quant_developer: pass (2/2 checks)

### D-012: Fix Environment Action Flow (L-007)
- **Date**: 2026-02-02
- **Decided By**: @rl_engineer (autonomous)
- **Trigger**: L-007 bloqueante - Training genera 0 trades
- **Choice**: Investigar y corregir `gym_env/nautilus_env.py`
- **Scope**: Arquitectura de comunicación env.step() → GymTradingStrategy
- **Priority**: CRITICAL - sin esto no hay training funcional

---

### D-011: Train Validation Test Agent
- **Date**: 2026-02-02
- **Decided By**: @rl_engineer (autonomous)
- **Trigger**: test_agent_001 failed validation (0 trades, trained with only 100 steps)
- **Choice**: Train new agent with 100K timesteps for pipeline validation test
- **Rationale**:
  - Config says 1M steps for production, but 100K sufficient for pipeline test
  - DQ-003 allows local training for <10 agents
  - Need to validate that training → validation flow works
- **Config Applied**:
  - Symbol: SPY.NASDAQ (has data in catalog)
  - Algorithm: PPO
  - Timesteps: 100,000 (test), then scale to 1M
- **Expected Outcome**: Agent generates trades, passes or fails Filter 1 with real metrics
- **Next Step**: If pipeline works, train batch of 10 agents with 1M steps each

---


### CHG-20260202-094804: 1 files changed: code (1 files)
- **Date**: 2026-02-02 09:48
- **Proposed By**: @rl_engineer
- **Validated By**: @rl_engineer, @quant_developer
- **Status**: AUTO-APPROVED
- **Files**: gym_env/nautilus_env.py
- **Validations**:
  - rl_engineer: pass (2/2 checks)
  - quant_developer: pass (2/2 checks)


### CHG-20260202-095050: 6 files changed: code (5 files)
- **Date**: 2026-02-02 09:50
- **Proposed By**: @rl_engineer
- **Validated By**: @quant_developer, @rl_engineer, @mlops_engineer
- **Status**: AUTO-APPROVED
- **Files**: .roles/DECISIONS.md, .roles/LESSONS_LEARNED.md, .roles/rl_engineer/STATE.md, gym_env/nautilus_env.py, training/train_agent.py...
- **Validations**:
  - quant_developer: pass (2/2 checks)
  - rl_engineer: pass (2/2 checks)
  - mlops_engineer: pass (2/2 checks)


### CHG-20260202-095109: 6 files changed: code (5 files)
- **Date**: 2026-02-02 09:51
- **Proposed By**: @rl_engineer
- **Validated By**: @mlops_engineer, @rl_engineer, @quant_developer
- **Status**: AUTO-APPROVED
- **Files**: .roles/DECISIONS.md, .roles/LESSONS_LEARNED.md, .roles/rl_engineer/STATE.md, gym_env/nautilus_env.py, training/train_agent.py...
- **Validations**:
  - mlops_engineer: pass (2/2 checks)
  - rl_engineer: pass (2/2 checks)
  - quant_developer: pass (2/2 checks)

## Template

```markdown
### D-XXX: [Title]
- **Date**: YYYY-MM-DD
- **Decided By**: @role(s)
- **Choice**: What was decided
- **Alternatives Considered**: What else was possible
- **Rationale**: Why this choice
- **Impact**: What code/process changed
- **Lessons**: What we learned (optional)
```
