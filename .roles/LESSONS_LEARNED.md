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

### L-007: Environment Episodes Terminate Immediately (RESOLVED)
- **Date**: 2026-02-02
- **Updated**: 2026-02-03 (FIX APPLIED)
- **Discovered By**: @rl_engineer (autonomous validation D-011)
- **Problem**: Episodes terminated after 1 step because `_bars_data` was empty
- **Symptoms**:
  - mean_ep_length: 1
  - mean_reward: 0
  - explained_variance: nan
- **RunPod Evidence** (5 hours, $1.00 spent):
  ```
  mean_ep_length: 1          # Episodes end after 1 step!
  mean_reward: 0             # Always zero reward
  ```
- **Root Cause Analysis**:
  1. `reset()` set `_current_bar_idx = lookback_period` (20)
  2. If catalog failed to load or bars query returned empty, `_bars_data = []`
  3. In `step()`: `if _current_bar_idx >= len(_bars_data)` → `20 >= 0` → True!
  4. Episode terminated immediately returning `done=True`
- **Solution Applied (2026-02-03)**:
  1. Added validation in `reset()` to raise `ValueError` if insufficient bars
  2. Improved `_load_catalog()` with path diagnostics and fallback paths
  3. Added logging for available instruments and loaded bar count
  4. Added test `test_l007_no_data_raises_error()` to verify fix
- **Files Changed**:
  - `gym_env/nautilus_env.py` - Added fail-fast validation
  - `tests/test_gym_env.py` - Fixed imports + added L-007 test
- **Status**: **RESOLVED** - Fix applied, tests pass
- **Prevention**: Always validate data is loaded before starting episode

### L-008: Tar.gz Structure Must Preserve Directories
- **Date**: 2026-02-02
- **Discovered By**: @mlops_engineer
- **Problem**: `code.tar.gz` had all files in root instead of subdirectories, causing `ModuleNotFoundError: No module named 'gym_env'`
- **Symptoms**:
  - Pods downloaded code but training failed immediately
  - Python couldn't find modules despite files existing
- **Root Cause**: Tar was created from wrong directory or with wrong arguments
- **Solution**:
  ```bash
  cd /project/root && tar -czvf code.tar.gz gym_env/ training/ strategies/ ...
  ```
- **Prevention**:
  - Always verify structure: `tar -tzf code.tar.gz | head -20`
  - Should show `gym_env/`, `gym_env/__init__.py`, not `__init__.py` at root
- **Files Affected**: VPS `/var/www/html/nautilus/code.tar.gz`

### L-009: RunPod SSH Requires Account-Level Key Setup
- **Date**: 2026-02-02
- **Discovered By**: @mlops_engineer (with user)
- **Problem**: SSH to pods asked for password despite `PUBLIC_KEY` env var
- **Symptoms**: `Permission denied (publickey,password)`
- **Root Cause**: RunPod requires SSH key in account settings, not just per-pod env
- **Solution**:
  1. Go to RunPod dashboard → Pod → Connect tab
  2. Paste public key in "SSH public key" field
  3. Click Save
  4. Create NEW pod (existing pods don't get updated key)
- **Prevention**: Set up SSH key in RunPod account BEFORE creating pods
- **Files Affected**: RunPod account settings

### L-010: Cross-Platform Parquet Serialization Incompatibility (CRITICAL)
- **Date**: 2026-02-03
- **Discovered By**: @rl_engineer, @mlops_engineer
- **Problem**: NautilusTrader ParquetDataCatalog created on Windows cannot be read on Linux
- **Symptoms**:
  ```
  Price raw bytes must be exactly the size of PriceRaw: TryFromSliceError(())
  ```
  - Catalog loads (instruments visible)
  - Bars query crashes with Rust panic
  - Results in L-007 symptoms (ep_length=1, reward=0)
- **Root Cause**: Binary serialization of Price/Quantity types differs between Windows and Linux
- **Solution**:
  1. Keep raw data (CSV) on PC or VPS
  2. Create catalog IN LINUX (VPS or RunPod pod)
  3. Use `scripts/create_catalog_linux.py` to convert CSV → Catalog
  4. Sync Linux-native catalog to RunPod for training
- **Correct Data Flow**:
  ```
  CSV (any OS) → Catalog (create in Linux) → Training (Linux/RunPod)
  ```
- **Files Created**:
  - `scripts/create_catalog_linux.py` - Converts CSV to catalog
  - `pod_startup.sh` on VPS - Downloads raw data and creates catalog on pod
- **Verification**: After fix, training showed:
  - ep_len_mean: 308 (was 1)
  - n_updates: 190 (was 0)
  - PPO learning confirmed
- **Prevention**: NEVER create catalogs on Windows for Linux use
- **Cost of Discovery**: ~$1.50 in RunPod GPU time

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
