# nautilus-rl

Reinforcement Learning trading environment with a controlled training and evaluation protocol.

## Overview
This repository contains a custom Gym-compatible trading environment and a full experimental
protocol designed to evaluate whether an RL agent learns a **coherent trading policy**, not just
profit by chance.

The focus is on:
- clean environment design
- strict separation of train / validation / test data
- multi-seed training
- comparison against deterministic and random baselines

Profitability is **not** the primary objective at this stage.

## Key features
- Discrete action space: HOLD / CLOSE / LONG / SHORT
- Dense, stable reward (SIMPLE_V1)
- Clean observation space (36 features, no constants)
- Deterministic MA20/MA50 baseline
- Reproducible training protocol with early stopping
- Automatic benchmark report with PASS / FAIL verdict

## Main script
Run the full training + evaluation protocol:

```bash
python scripts/train_and_evaluate.py \
  --catalog-path data/catalog \
  --instrument-id BTCUSDT.BINANCE \
  --venue BINANCE \
  --seeds 42 123 456 789 1024 \
  --timesteps 500000 \
  --max-steps 252 \
  --output benchmark_results/
```
