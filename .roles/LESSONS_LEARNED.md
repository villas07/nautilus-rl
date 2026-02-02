# Lessons Learned

Continuous improvement through documented learnings.

---

## API & Compatibility

### L-001: NautilusTrader API Changes Frequently
- **Date**: 2026-02-02
- **Discovered By**: @rl_engineer
- **Problem**: `MarginAccount.equity()` method didn't exist
- **Solution**: Use `balance_total(currency).as_double()`
- **Prevention**: Always check API docs for current version
- **Files Affected**: `gym_env/nautilus_env.py`

### L-002: Precision Mismatch Causes Errors
- **Date**: 2026-02-02
- **Discovered By**: @quant_developer
- **Problem**: Bar prices had 8 decimals, instrument expected 2
- **Solution**: Match bar precision to instrument `price_precision`
- **Prevention**: Always verify precision when writing to catalog
- **Files Affected**: `scripts/populate_catalog_standalone.py`

### L-003: BacktestVenueConfig API Changed
- **Date**: 2026-02-02
- **Discovered By**: @rl_engineer
- **Problem**: `maker_fee`/`taker_fee` not valid kwargs
- **Solution**: Fees are now set on instrument level, not venue
- **Prevention**: Check NautilusTrader changelog for breaking changes

### L-004: Decimal vs Float for Leverage
- **Date**: 2026-02-02
- **Discovered By**: @rl_engineer
- **Problem**: `default_leverage` must be `Decimal`, not `float`
- **Solution**: `Decimal(str(value))` conversion
- **Prevention**: Use type hints, check error messages carefully

---

## Development Process

### L-005: Install Dependencies Incrementally
- **Date**: 2026-02-02
- **Discovered By**: @mlops_engineer
- **Problem**: Missing tensorboard, tqdm, rich caused training failures
- **Solution**: Test imports before running full pipeline
- **Prevention**: Add `scripts/check_dependencies.py`

---

## Architecture

### L-006: Keep Strategy Params Separate from Config
- **Date**: 2026-02-02
- **Discovered By**: @rl_engineer
- **Problem**: `StrategyConfig` doesn't accept custom params
- **Solution**: Create separate dataclass (e.g., `GymStrategyParams`)
- **Prevention**: Document NautilusTrader patterns

---


### L-AUTO-CHG-20260202-090352: Change Rejected by Governance
- **Date**: 2026-02-02
- **Change ID**: CHG-20260202-090352
- **Proposed By**: @rl_engineer
- **Description**: 1 files changed: code (1 files)
- **Failed Validations**:
  - **rl_engineer**: Unit tests failed: 
    - FAILED: unit_tests_pass
- **Action Taken**: Auto-reverted
- **Prevention**: Review validation criteria before proposing similar changes


### L-AUTO-CHG-20260202-090515: Change Rejected by Governance
- **Date**: 2026-02-02
- **Change ID**: CHG-20260202-090515
- **Proposed By**: @mlops_engineer
- **Description**: 6 files changed: code (5 files)
- **Failed Validations**:
  - **rl_engineer**: Unit tests failed: 
    - FAILED: unit_tests_pass
- **Action Taken**: Auto-reverted
- **Prevention**: Review validation criteria before proposing similar changes


### L-AUTO-CHG-20260202-091916: Change Rejected by Governance
- **Date**: 2026-02-02
- **Change ID**: CHG-20260202-091916
- **Proposed By**: @quant_developer
- **Description**: 6 files changed: code (3 files)
- **Failed Validations**:
  - **quant_developer**: NautilusTrader imports OK, NautilusTrader compatibility verified, Unit tests failed: C:\Users\PcVIP\AppData\Local\Programs\Python\Python311\python.exe: No module named pytest

    - FAILED: unit_tests_pass
- **Action Taken**: Auto-reverted
- **Prevention**: Review validation criteria before proposing similar changes

### L-007: Environment-Strategy Action Flow Broken
- **Date**: 2026-02-02
- **Discovered By**: @rl_engineer (autonomous validation D-011)
- **Problem**: NautilusBacktestEnv.step() doesn't route RL actions to GymTradingStrategy
- **Symptoms**:
  - Backtest completes with Total orders: 0
  - 250 iterations (bars) process instantly (~100ms)
  - No PPO training metrics appear
  - Agent never generates trades
- **Root Cause**: The GymTradingStrategy receives bar updates but never receives action signals from the RL agent. The step() method advances the backtest but doesn't execute trades based on actions.
- **Investigation Needed**:
  1. How does step() communicate actions to the strategy?
  2. Does on_bar() in GymTradingStrategy check for external action signals?
  3. Is there a missing callback or event mechanism?
- **Files Affected**: `gym_env/nautilus_env.py` (NautilusBacktestEnv class)
- **Status**: OPEN - requires architecture review

---

## Template for New Lessons

```markdown
### L-XXX: [Short Title]
- **Date**: YYYY-MM-DD
- **Discovered By**: @role
- **Problem**: What went wrong
- **Solution**: How it was fixed
- **Prevention**: How to avoid in future
- **Files Affected**: Which files changed
```

---

## Statistics

| Category | Count |
|----------|-------|
| API/Compatibility | 4 |
| Development Process | 1 |
| Architecture | 1 |
| **Total** | **6** |

_Last updated: 2026-02-02_
