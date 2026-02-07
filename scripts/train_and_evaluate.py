#!/usr/bin/env python3
"""
Training and Evaluation Protocol — Ticket 5

Orchestrates controlled training with multi-seed, early stopping,
and comparison against deterministic baseline to decide if the agent
learns a coherent policy or just noise.

Usage:
    python scripts/train_and_evaluate.py \
        --catalog-path data/catalog \
        --instrument-id BTCUSDT.BINANCE \
        --venue BINANCE \
        --seeds 42 123 456 789 1024 \
        --timesteps 500000 \
        --max-steps 252 \
        --output benchmark_results/
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_total_bars(catalog_path: str, instrument_id: str, start_date: str, end_date: str) -> int:
    """Load catalog and return total number of bars available."""
    import pandas as pd
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.model.identifiers import InstrumentId

    catalog = ParquetDataCatalog(catalog_path)
    iid = InstrumentId.from_str(instrument_id)
    bars = catalog.bars(
        instrument_ids=[str(iid)],
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
    )
    n = len(list(bars)) if bars else 0
    return n


def compute_splits(n_bars: int):
    """Compute train/val/test splits (70/15/15)."""
    train_end = int(n_bars * 0.70)
    val_end = int(n_bars * 0.85)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_bars),
        "total": n_bars,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(
    seed: int,
    config_kwargs: Dict[str, Any],
    train_split: tuple,
    val_split: tuple,
    total_timesteps: int,
    output_dir: Path,
    policy_mode: str = "baseline",
) -> Optional[Path]:
    """
    Train PPO with a single seed, using EvalCallback + early stopping.

    Args:
        policy_mode: "baseline" (4-action) or "baseline_plus_rl_exit" (2-action exit gate).

    Returns path to best model or None if training failed.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        StopTrainingOnNoModelImprovement,
    )

    from gym_env import NautilusGymEnv, NautilusEnvConfig

    seed_dir = output_dir / f"seed_{seed}"
    best_dir = seed_dir / "best"
    checkpoint_dir = seed_dir / "checkpoints"
    log_dir = seed_dir / "logs"
    for d in [best_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Seed {seed}] Starting training", timesteps=total_timesteps)

    # --- Train env ---
    train_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=train_split[0],
        bar_end_idx=train_split[1],
    )
    train_env = NautilusGymEnv(config=train_config)

    # --- Val env ---
    val_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=val_split[0],
        bar_end_idx=val_split[1],
    )
    val_env = NautilusGymEnv(config=val_config)

    # Wrap in ExitGateEnv if using exit gate mode
    if policy_mode == "baseline_plus_rl_exit":
        from gym_env.exit_gate_env import ExitGateEnv
        from scripts.evaluate_policies import DeterministicBaseline
        train_env = ExitGateEnv(train_env, DeterministicBaseline())
        val_env = ExitGateEnv(val_env, DeterministicBaseline())

    # --- Callbacks ---
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_seed{seed}",
        verbose=0,
    )

    # --- PPO ---
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            tensorboard_log=str(log_dir),
        )

        t0 = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, checkpoint_cb],
            progress_bar=True,
        )
        elapsed = time.time() - t0
        logger.info(f"[Seed {seed}] Training done", elapsed_s=f"{elapsed:.1f}")

    except Exception as e:
        logger.error(f"[Seed {seed}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        train_env.close()
        val_env.close()
        return None
    finally:
        train_env.close()
        val_env.close()

    # Return best model path
    best_model_path = best_dir / "best_model.zip"
    if best_model_path.exists():
        logger.info(f"[Seed {seed}] Best model at {best_model_path}")
        return best_model_path

    # Fallback: save final model
    fallback_path = best_dir / "best_model.zip"
    model.save(str(fallback_path))
    logger.warning(f"[Seed {seed}] No best model found, saved final model as fallback")
    return fallback_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model_path: Path,
    seed: int,
    config_kwargs: Dict[str, Any],
    test_split: tuple,
    policy_mode: str = "baseline",
) -> Dict[str, Any]:
    """Evaluate a trained model on the test set."""
    from stable_baselines3 import PPO

    from gym_env import NautilusGymEnv, NautilusEnvConfig

    # Import policies from evaluate_policies
    from scripts.evaluate_policies import (
        DeterministicBaseline,
        RandomPolicy,
        MLPolicy,
        evaluate_policy,
    )

    test_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=test_split[0],
        bar_end_idx=test_split[1],
    )

    results = {}

    # --- ML model ---
    try:
        model = PPO.load(str(model_path))
        ml_policy = MLPolicy(model)
        env = NautilusGymEnv(config=test_config)
        if policy_mode == "baseline_plus_rl_exit":
            from gym_env.exit_gate_env import ExitGateEnv
            env = ExitGateEnv(env, DeterministicBaseline())
        ml_result = evaluate_policy(ml_policy, env, n_episodes=1)
        ml_result["seed"] = seed
        results["ml"] = ml_result
        env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] ML eval failed: {e}")
        results["ml"] = {"policy": "ml_agent", "seed": seed, "error": str(e)}

    # --- Baseline ---
    try:
        baseline_policy = DeterministicBaseline()
        env = NautilusGymEnv(config=test_config)
        baseline_result = evaluate_policy(baseline_policy, env, n_episodes=1)
        results["baseline"] = baseline_result
        env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] Baseline eval failed: {e}")
        results["baseline"] = {"policy": "baseline_ma_cross", "error": str(e)}

    # --- Random ---
    try:
        random_policy = RandomPolicy()
        env = NautilusGymEnv(config=test_config)
        if policy_mode == "baseline_plus_rl_exit":
            from gym_env.exit_gate_env import ExitGateEnv
            env = ExitGateEnv(env, DeterministicBaseline())
        random_result = evaluate_policy(random_policy, env, n_episodes=1)
        results["random"] = random_result
        env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] Random eval failed: {e}")
        results["random"] = {"policy": "random", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Acceptance Criteria
# ---------------------------------------------------------------------------

def apply_acceptance_criteria(
    seed_results: List[Dict[str, Any]],
    policy_mode: str = "baseline",
) -> Dict[str, Any]:
    """
    Apply acceptance criteria across all seeds.

    Criteria (baseline mode):
    1. NO over-trading: num_trades <= baseline_num_trades * 2
    2. max_drawdown <= baseline_max_drawdown * 1.2
    3. pct_time_flat between 5% and 95%
    4. Stability: std(total_return) < mean(total_return) * 2

    Criteria (baseline_plus_rl_exit mode — stricter on risk):
    1. num_trades <= baseline_num_trades * 1.2
    2. max_drawdown <= baseline_max_drawdown * 0.9  (must improve DD)
    3. pct_time_flat between 5% and 95%
    4. Stability: std(total_return) < mean(total_return) * 2
    """
    warnings = []
    ml_returns = []
    ml_trades = []
    ml_drawdowns = []
    ml_flat_pcts = []

    # Collect ML metrics from all seeds
    for sr in seed_results:
        ml = sr.get("ml", {})
        if "error" in ml:
            continue
        ml_returns.append(ml.get("total_return", 0.0))
        ml_trades.append(ml.get("num_trades", 0))
        ml_drawdowns.append(ml.get("max_drawdown", 0.0))
        ml_flat_pcts.append(ml.get("pct_time_flat", 50.0))

    if not ml_returns:
        return {
            "verdict": "FAIL",
            "reason": "No valid ML results from any seed",
            "warnings": ["NO_VALID_RESULTS"],
        }

    # Use first seed's baseline as reference (same test set)
    baseline = seed_results[0].get("baseline", {})
    baseline_trades = baseline.get("num_trades", 1)
    baseline_dd = baseline.get("max_drawdown", 0.01)

    mean_return = float(np.mean(ml_returns))
    std_return = float(np.std(ml_returns))
    mean_trades = float(np.mean(ml_trades))
    mean_dd = float(np.mean(ml_drawdowns))
    mean_flat = float(np.mean(ml_flat_pcts))

    # --- Criteria checks (thresholds depend on policy mode) ---
    if policy_mode == "baseline_plus_rl_exit":
        trade_mult = 1.2   # Max 20% more trades
        dd_mult = 0.9      # Must IMPROVE drawdown by 10%
    else:
        trade_mult = 2.0
        dd_mult = 1.2

    # 1. Over-trading
    if mean_trades > baseline_trades * trade_mult:
        warnings.append("OVER-TRADING")

    # 2. Drawdown
    if mean_dd > baseline_dd * dd_mult:
        warnings.append("HIGH-DRAWDOWN")

    # 3. Flat time
    if mean_flat < 5.0:
        warnings.append("NEVER-FLAT")
    if mean_flat > 95.0:
        warnings.append("ALWAYS-FLAT")

    # 4. Stability across seeds
    if len(ml_returns) > 1 and abs(mean_return) > 0:
        if std_return > abs(mean_return) * 2:
            warnings.append("UNSTABLE")

    # --- Verdict ---
    critical_warnings = {"OVER-TRADING", "HIGH-DRAWDOWN", "UNSTABLE", "ALWAYS-FLAT", "NEVER-FLAT"}
    has_critical = bool(set(warnings) & critical_warnings)
    verdict = "FAIL" if has_critical else "PASS"

    reasons = []
    if "OVER-TRADING" in warnings:
        reasons.append(f"Avg trades ({mean_trades:.0f}) > {trade_mult}x baseline ({baseline_trades:.0f})")
    if "HIGH-DRAWDOWN" in warnings:
        reasons.append(f"Avg DD ({mean_dd*100:.1f}%) > {dd_mult}x baseline ({baseline_dd*100:.1f}%)")
    if "UNSTABLE" in warnings:
        reasons.append(f"Return std ({std_return*100:.2f}%) > 2x mean ({mean_return*100:.2f}%)")
    if "ALWAYS-FLAT" in warnings:
        reasons.append(f"Avg flat time ({mean_flat:.1f}%) > 95%")
    if "NEVER-FLAT" in warnings:
        reasons.append(f"Avg flat time ({mean_flat:.1f}%) < 5%")

    return {
        "verdict": verdict,
        "reasons": reasons if reasons else ["All criteria passed"],
        "warnings": warnings,
        "stats": {
            "mean_return": mean_return,
            "std_return": std_return,
            "mean_trades": mean_trades,
            "mean_drawdown": mean_dd,
            "mean_flat_pct": mean_flat,
            "baseline_trades": baseline_trades,
            "baseline_drawdown": baseline_dd,
            "n_seeds_valid": len(ml_returns),
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    seed_results: List[Dict[str, Any]],
    criteria: Dict[str, Any],
    splits: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate human-readable comparison report."""
    lines = [
        "=" * 90,
        "TRAINING & EVALUATION PROTOCOL REPORT",
        f"Generated: {datetime.now().isoformat()}",
        f"Instrument: {config.get('instrument_id', 'N/A')}",
        f"Total bars: {splits['total']} | "
        f"Train: {splits['train'][0]}-{splits['train'][1]} ({splits['train'][1]-splits['train'][0]}) | "
        f"Val: {splits['val'][0]}-{splits['val'][1]} ({splits['val'][1]-splits['val'][0]}) | "
        f"Test: {splits['test'][0]}-{splits['test'][1]} ({splits['test'][1]-splits['test'][0]})",
        "=" * 90,
        "",
        f"{'Policy':<20} {'Seed':>5} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} "
        f"{'AvgHold':>8} {'Flat%':>7} {'MeanRwd':>9}",
        "-" * 90,
    ]

    # Baseline row (same across seeds, take first)
    bl = seed_results[0].get("baseline", {})
    if "error" not in bl:
        lines.append(
            f"{'baseline_ma_cross':<20} {'  ---':>5} "
            f"{bl.get('total_return', 0)*100:>8.2f}% "
            f"{bl.get('max_drawdown', 0)*100:>7.2f}% "
            f"{bl.get('num_trades', 0):>7.0f} "
            f"{bl.get('avg_hold_duration', 0):>8.1f} "
            f"{bl.get('pct_time_flat', 0):>6.1f}% "
            f"{bl.get('mean_reward', 0):>9.4f}"
        )

    # Random row (same across seeds, take first)
    rnd = seed_results[0].get("random", {})
    if "error" not in rnd:
        lines.append(
            f"{'random':<20} {'  ---':>5} "
            f"{rnd.get('total_return', 0)*100:>8.2f}% "
            f"{rnd.get('max_drawdown', 0)*100:>7.2f}% "
            f"{rnd.get('num_trades', 0):>7.0f} "
            f"{rnd.get('avg_hold_duration', 0):>8.1f} "
            f"{rnd.get('pct_time_flat', 0):>6.1f}% "
            f"{rnd.get('mean_reward', 0):>9.4f}"
        )

    lines.append("-" * 90)

    # ML rows per seed
    for sr in seed_results:
        ml = sr.get("ml", {})
        if "error" in ml:
            lines.append(f"{'ml_agent':<20} {ml.get('seed', '?'):>5} {'ERROR':>9}")
            continue
        lines.append(
            f"{'ml_agent':<20} {ml.get('seed', '?'):>5} "
            f"{ml.get('total_return', 0)*100:>8.2f}% "
            f"{ml.get('max_drawdown', 0)*100:>7.2f}% "
            f"{ml.get('num_trades', 0):>7.0f} "
            f"{ml.get('avg_hold_duration', 0):>8.1f} "
            f"{ml.get('pct_time_flat', 0):>6.1f}% "
            f"{ml.get('mean_reward', 0):>9.4f}"
        )

    # ML mean row
    stats = criteria.get("stats", {})
    if stats.get("n_seeds_valid", 0) > 0:
        lines.append("-" * 90)
        lines.append(
            f"{'ml_agent (MEAN)':<20} {'  AVG':>5} "
            f"{stats['mean_return']*100:>8.2f}% "
            f"{stats['mean_drawdown']*100:>7.2f}% "
            f"{stats['mean_trades']:>7.0f} "
            f"{'---':>8} "
            f"{stats['mean_flat_pct']:>6.1f}% "
            f"{'---':>9}"
        )
        lines.append(
            f"{'ml_agent (STD)':<20} {'  STD':>5} "
            f"{stats['std_return']*100:>8.2f}%"
        )

    lines.append("")
    lines.append("=" * 90)

    # Warnings
    lines.append("FLAGS:")
    if criteria["warnings"]:
        for w in criteria["warnings"]:
            lines.append(f"  [!] {w}")
    else:
        lines.append("  None")

    # Verdict
    lines.append("")
    lines.append(f"VERDICT: {criteria['verdict']}")
    for reason in criteria.get("reasons", []):
        lines.append(f"  - {reason}")

    lines.append("")
    lines.append("=" * 90)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Training and Evaluation Protocol (Ticket 5)"
    )
    parser.add_argument(
        "--catalog-path", type=str, default="data/catalog",
        help="Path to ParquetDataCatalog",
    )
    parser.add_argument(
        "--instrument-id", type=str, default="BTCUSDT.BINANCE",
        help="Instrument ID",
    )
    parser.add_argument(
        "--venue", type=str, default="BINANCE",
        help="Venue name",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
        help="Seeds for multi-seed training",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total timesteps per seed",
    )
    parser.add_argument(
        "--max-steps", type=int, default=252,
        help="Max steps per episode (252 = ~1 year daily)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01",
        help="Data start date",
    )
    parser.add_argument(
        "--end-date", type=str, default="2025-12-31",
        help="Data end date",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--policy-mode", type=str,
        choices=["baseline", "baseline_plus_rl_exit"],
        default="baseline",
        help="Policy mode: baseline (4-action) or baseline_plus_rl_exit (2-action exit gate)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING & EVALUATION PROTOCOL — mode: {args.policy_mode}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load data and compute splits
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data and computing splits...")

    n_bars = load_total_bars(
        args.catalog_path, args.instrument_id,
        args.start_date, args.end_date,
    )

    if n_bars == 0:
        print(f"ERROR: No bars found for {args.instrument_id} in {args.catalog_path}")
        sys.exit(1)

    splits = compute_splits(n_bars)
    print(f"  Total bars: {n_bars}")
    print(f"  Train: bars[{splits['train'][0]}:{splits['train'][1]}] "
          f"({splits['train'][1] - splits['train'][0]} bars)")
    print(f"  Val:   bars[{splits['val'][0]}:{splits['val'][1]}] "
          f"({splits['val'][1] - splits['val'][0]} bars)")
    print(f"  Test:  bars[{splits['test'][0]}:{splits['test'][1]}] "
          f"({splits['test'][1] - splits['test'][0]} bars)")

    # Common env config kwargs (without split indices)
    config_kwargs = {
        "instrument_id": args.instrument_id,
        "venue": args.venue,
        "catalog_path": args.catalog_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "max_episode_steps": args.max_steps,
        "reward_type": "simple_v1",
    }

    output_dir = Path(args.output) / "protocol"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: Train with multiple seeds
    # ------------------------------------------------------------------
    print(f"\n[2/5] Training with {len(args.seeds)} seeds: {args.seeds}")
    print(f"  Timesteps per seed: {args.timesteps}")

    best_models = {}
    for seed in args.seeds:
        print(f"\n--- Training seed {seed} ---")
        model_path = train_seed(
            seed=seed,
            config_kwargs=config_kwargs,
            train_split=splits["train"],
            val_split=splits["val"],
            total_timesteps=args.timesteps,
            output_dir=output_dir,
            policy_mode=args.policy_mode,
        )
        if model_path:
            best_models[seed] = model_path
            print(f"  Seed {seed}: best model at {model_path}")
        else:
            print(f"  Seed {seed}: FAILED")

    if not best_models:
        print("\nERROR: All seeds failed. Aborting.")
        sys.exit(1)

    print(f"\n  Successful seeds: {len(best_models)}/{len(args.seeds)}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate each seed on test set
    # ------------------------------------------------------------------
    print(f"\n[3/5] Evaluating on test set (bars[{splits['test'][0]}:{splits['test'][1]}])...")

    seed_results = []
    for seed, model_path in best_models.items():
        print(f"\n  Evaluating seed {seed}...")
        result = evaluate_on_test(
            model_path=model_path,
            seed=seed,
            config_kwargs=config_kwargs,
            test_split=splits["test"],
            policy_mode=args.policy_mode,
        )
        seed_results.append(result)

        ml = result.get("ml", {})
        if "error" not in ml:
            print(f"    ML Return: {ml.get('total_return', 0)*100:.2f}% | "
                  f"Trades: {ml.get('num_trades', 0):.0f} | "
                  f"MaxDD: {ml.get('max_drawdown', 0)*100:.2f}%")
        else:
            print(f"    ML: ERROR - {ml.get('error', 'unknown')}")

    # ------------------------------------------------------------------
    # Step 4: Apply acceptance criteria
    # ------------------------------------------------------------------
    print("\n[4/5] Applying acceptance criteria...")

    criteria = apply_acceptance_criteria(seed_results, policy_mode=args.policy_mode)

    print(f"  Verdict: {criteria['verdict']}")
    for w in criteria.get("warnings", []):
        print(f"  [!] {w}")
    for r in criteria.get("reasons", []):
        print(f"  - {r}")

    # ------------------------------------------------------------------
    # Step 5: Generate report and save
    # ------------------------------------------------------------------
    print("\n[5/5] Generating report...")

    report = generate_report(
        seed_results=seed_results,
        criteria=criteria,
        splits=splits,
        config={"instrument_id": args.instrument_id, "venue": args.venue},
    )
    print(f"\n{report}")

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.output) / f"protocol_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "instrument_id": args.instrument_id,
            "venue": args.venue,
            "catalog_path": args.catalog_path,
            "seeds": args.seeds,
            "timesteps": args.timesteps,
            "max_steps": args.max_steps,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "policy_mode": args.policy_mode,
        },
        "splits": splits,
        "seed_results": seed_results,
        "acceptance_criteria": criteria,
        "verdict": criteria["verdict"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {json_path}")
    print(f"\nDone. Verdict: {criteria['verdict']}")


if __name__ == "__main__":
    main()
