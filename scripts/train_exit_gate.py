#!/usr/bin/env python3
"""
Exit Gate Training & Evaluation — Ticket 6

Trains a PPO agent that only decides WHEN to exit positions (HOLD vs CLOSE),
while a deterministic MA20/MA50 baseline controls entries.

Compares three strategies on the test set:
  1. Baseline puro (MA20/MA50 cross, 4-action env)
  2. Baseline + RL Exit Gate (baseline entries, RL exits, 2-action env)
  3. Random Exit (baseline entries, random exits, 2-action env)

Usage:
    python scripts/train_exit_gate.py \\
        --catalog-path data/catalog \\
        --instrument-id BTCUSDT.BINANCE \\
        --venue BINANCE \\
        --seeds 42 \\
        --timesteps 50000 \\
        --max-steps 252 \\
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

# Reuse helpers from Ticket 5 (NO modification)
from scripts.train_and_evaluate import load_total_bars, compute_splits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_exit_gate_seed(
    seed: int,
    config_kwargs: Dict[str, Any],
    train_split: tuple,
    val_split: tuple,
    total_timesteps: int,
    output_dir: Path,
) -> Optional[Path]:
    """
    Train PPO on ExitGateEnv (2 actions) with a single seed.

    Returns path to best model or None if training failed.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        StopTrainingOnNoModelImprovement,
    )

    from gym_env import NautilusGymEnv, NautilusEnvConfig
    from gym_env.exit_gate_env import ExitGateEnv
    from scripts.evaluate_policies import DeterministicBaseline

    seed_dir = output_dir / f"seed_{seed}"
    best_dir = seed_dir / "best"
    checkpoint_dir = seed_dir / "checkpoints"
    log_dir = seed_dir / "logs"
    for d in [best_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Seed {seed}] Starting Exit Gate training", timesteps=total_timesteps)

    # --- Train env (ExitGateEnv) ---
    train_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=train_split[0],
        bar_end_idx=train_split[1],
    )
    train_inner = NautilusGymEnv(config=train_config)
    train_env = ExitGateEnv(train_inner, DeterministicBaseline())

    # --- Val env (ExitGateEnv) ---
    val_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=val_split[0],
        bar_end_idx=val_split[1],
    )
    val_inner = NautilusGymEnv(config=val_config)
    val_env = ExitGateEnv(val_inner, DeterministicBaseline())

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
        name_prefix=f"exit_gate_seed{seed}",
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
# Evaluation helpers
# ---------------------------------------------------------------------------

class ExitGateMLPolicy:
    """Wraps a trained SB3 model for use inside ExitGateEnv."""

    name = "baseline_rl_exit"

    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, obs: np.ndarray, env) -> int:
        """Predict action (0=HOLD, 1=CLOSE) for ExitGateEnv."""
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class ExitGateRandomPolicy:
    """Random exit policy for ExitGateEnv (Discrete(2))."""

    name = "random_exit"

    def reset(self):
        pass

    def act(self, obs: np.ndarray, env) -> int:
        return int(env.action_space.sample())


def evaluate_exit_gate_policy(policy, env, n_episodes: int = 1) -> Dict[str, Any]:
    """
    Run a policy through an ExitGateEnv and compute metrics.

    Compatible with the metrics format from evaluate_policies.evaluate_policy().
    """
    ep_returns = []
    ep_drawdowns = []
    ep_num_trades = []
    ep_hold_durations = []
    ep_turnovers = []
    ep_flat_pcts = []
    ep_rewards = []

    for ep in range(n_episodes):
        policy.reset()
        obs, info = env.reset()

        total_reward = 0.0
        steps = 0
        flat_steps = 0
        abs_position_changes = 0.0
        prev_pos = 0.0

        while True:
            action = policy.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Track flat time (access inner env position)
            current_pos = env.env._position if hasattr(env, 'env') else env._position
            if current_pos == 0:
                flat_steps += 1

            # Track turnover
            abs_position_changes += abs(current_pos - prev_pos)
            prev_pos = current_pos

            if terminated or truncated:
                break

        # Collect episode-level stats
        ep_return = info.get("total_return", 0.0)
        ep_returns.append(ep_return)

        ep_stats = info.get("episode_stats", {})
        ep_drawdowns.append(ep_stats.get("max_drawdown", 0.0))
        ep_num_trades.append(ep_stats.get("num_trades", 0))
        ep_hold_durations.append(ep_stats.get("avg_hold_duration", 0.0))

        turnover = abs_position_changes / steps if steps > 0 else 0.0
        ep_turnovers.append(turnover)

        flat_pct = (flat_steps / steps * 100) if steps > 0 else 100.0
        ep_flat_pcts.append(flat_pct)

        ep_rewards.append(total_reward)

    return {
        "policy": policy.name,
        "n_episodes": n_episodes,
        "total_return": float(np.mean(ep_returns)),
        "total_return_std": float(np.std(ep_returns)),
        "max_drawdown": float(np.mean(ep_drawdowns)),
        "num_trades": float(np.mean(ep_num_trades)),
        "avg_hold_duration": float(np.mean(ep_hold_durations)),
        "turnover": float(np.mean(ep_turnovers)),
        "pct_time_flat": float(np.mean(ep_flat_pcts)),
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
    }


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test_exit_gate(
    model_path: Path,
    seed: int,
    config_kwargs: Dict[str, Any],
    test_split: tuple,
) -> Dict[str, Any]:
    """
    Evaluate on test set:
      1. Baseline puro (4-action env, DeterministicBaseline)
      2. Baseline + RL Exit (ExitGateEnv, trained model)
      3. Random Exit (ExitGateEnv, random 0/1)
    """
    from stable_baselines3 import PPO

    from gym_env import NautilusGymEnv, NautilusEnvConfig
    from gym_env.exit_gate_env import ExitGateEnv
    from scripts.evaluate_policies import (
        DeterministicBaseline,
        evaluate_policy,
    )

    test_config = NautilusEnvConfig(
        **config_kwargs,
        bar_start_idx=test_split[0],
        bar_end_idx=test_split[1],
    )

    results = {}

    # --- 1. Baseline puro (4-action env) ---
    try:
        baseline_policy = DeterministicBaseline()
        env = NautilusGymEnv(config=test_config)
        baseline_result = evaluate_policy(baseline_policy, env, n_episodes=1)
        results["baseline"] = baseline_result
        env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] Baseline eval failed: {e}")
        results["baseline"] = {"policy": "baseline_ma_cross", "error": str(e)}

    # --- 2. Baseline + RL Exit (ExitGateEnv) ---
    try:
        model = PPO.load(str(model_path))
        ml_exit_policy = ExitGateMLPolicy(model)
        inner_env = NautilusGymEnv(config=test_config)
        exit_env = ExitGateEnv(inner_env, DeterministicBaseline())
        ml_result = evaluate_exit_gate_policy(ml_exit_policy, exit_env, n_episodes=1)
        ml_result["seed"] = seed
        results["rl_exit"] = ml_result
        exit_env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] RL Exit eval failed: {e}")
        results["rl_exit"] = {"policy": "baseline_rl_exit", "seed": seed, "error": str(e)}

    # --- 3. Random Exit (ExitGateEnv) ---
    try:
        random_policy = ExitGateRandomPolicy()
        inner_env = NautilusGymEnv(config=test_config)
        exit_env = ExitGateEnv(inner_env, DeterministicBaseline())
        random_result = evaluate_exit_gate_policy(random_policy, exit_env, n_episodes=1)
        results["random_exit"] = random_result
        exit_env.close()
    except Exception as e:
        logger.error(f"[Seed {seed}] Random Exit eval failed: {e}")
        results["random_exit"] = {"policy": "random_exit", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_exit_gate_report(
    seed_results: List[Dict[str, Any]],
    splits: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate human-readable comparison report for Exit Gate evaluation."""
    lines = [
        "=" * 90,
        "EXIT GATE (HOLD/CLOSE) EVALUATION REPORT — Ticket 6",
        f"Generated: {datetime.now().isoformat()}",
        f"Instrument: {config.get('instrument_id', 'N/A')}",
        f"Total bars: {splits['total']} | "
        f"Train: {splits['train'][0]}-{splits['train'][1]} ({splits['train'][1]-splits['train'][0]}) | "
        f"Val: {splits['val'][0]}-{splits['val'][1]} ({splits['val'][1]-splits['val'][0]}) | "
        f"Test: {splits['test'][0]}-{splits['test'][1]} ({splits['test'][1]-splits['test'][0]})",
        "=" * 90,
        "",
        f"{'Policy':<25} {'Seed':>5} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} "
        f"{'AvgHold':>8} {'Flat%':>7} {'MeanRwd':>9}",
        "-" * 90,
    ]

    # Baseline row (same across seeds, take first)
    bl = seed_results[0].get("baseline", {})
    if "error" not in bl:
        lines.append(
            f"{'baseline_ma_cross':<25} {'  ---':>5} "
            f"{bl.get('total_return', 0)*100:>8.2f}% "
            f"{bl.get('max_drawdown', 0)*100:>7.2f}% "
            f"{bl.get('num_trades', 0):>7.0f} "
            f"{bl.get('avg_hold_duration', 0):>8.1f} "
            f"{bl.get('pct_time_flat', 0):>6.1f}% "
            f"{bl.get('mean_reward', 0):>9.4f}"
        )

    # Random Exit row (take first)
    rnd = seed_results[0].get("random_exit", {})
    if "error" not in rnd:
        lines.append(
            f"{'random_exit':<25} {'  ---':>5} "
            f"{rnd.get('total_return', 0)*100:>8.2f}% "
            f"{rnd.get('max_drawdown', 0)*100:>7.2f}% "
            f"{rnd.get('num_trades', 0):>7.0f} "
            f"{rnd.get('avg_hold_duration', 0):>8.1f} "
            f"{rnd.get('pct_time_flat', 0):>6.1f}% "
            f"{rnd.get('mean_reward', 0):>9.4f}"
        )

    lines.append("-" * 90)

    # RL Exit rows per seed
    for sr in seed_results:
        rl = sr.get("rl_exit", {})
        if "error" in rl:
            lines.append(f"{'baseline_rl_exit':<25} {rl.get('seed', '?'):>5} {'ERROR':>9}")
            continue
        lines.append(
            f"{'baseline_rl_exit':<25} {rl.get('seed', '?'):>5} "
            f"{rl.get('total_return', 0)*100:>8.2f}% "
            f"{rl.get('max_drawdown', 0)*100:>7.2f}% "
            f"{rl.get('num_trades', 0):>7.0f} "
            f"{rl.get('avg_hold_duration', 0):>8.1f} "
            f"{rl.get('pct_time_flat', 0):>6.1f}% "
            f"{rl.get('mean_reward', 0):>9.4f}"
        )

    # Mean across seeds
    valid_rl = [sr["rl_exit"] for sr in seed_results if "error" not in sr.get("rl_exit", {"error": True})]
    if len(valid_rl) > 1:
        mean_return = np.mean([r["total_return"] for r in valid_rl])
        std_return = np.std([r["total_return"] for r in valid_rl])
        mean_dd = np.mean([r["max_drawdown"] for r in valid_rl])
        mean_trades = np.mean([r["num_trades"] for r in valid_rl])
        mean_flat = np.mean([r["pct_time_flat"] for r in valid_rl])

        lines.append("-" * 90)
        lines.append(
            f"{'rl_exit (MEAN)':<25} {'  AVG':>5} "
            f"{mean_return*100:>8.2f}% "
            f"{mean_dd*100:>7.2f}% "
            f"{mean_trades:>7.0f} "
            f"{'---':>8} "
            f"{mean_flat:>6.1f}% "
            f"{'---':>9}"
        )
        lines.append(
            f"{'rl_exit (STD)':<25} {'  STD':>5} "
            f"{std_return*100:>8.2f}%"
        )

    lines.append("")
    lines.append("=" * 90)

    # Analysis vs baseline
    if bl and "error" not in bl and valid_rl:
        lines.append("")
        lines.append("ANALYSIS: RL Exit vs Baseline puro")
        lines.append("-" * 50)

        bl_return = bl.get("total_return", 0)
        bl_trades = bl.get("num_trades", 1)

        for rl in valid_rl:
            ret_diff = (rl["total_return"] - bl_return) * 100
            trade_ratio = rl["num_trades"] / bl_trades if bl_trades > 0 else float("inf")
            seed_label = rl.get("seed", "?")
            lines.append(
                f"  Seed {seed_label}: Return diff: {ret_diff:+.2f}pp | "
                f"Trade ratio: {trade_ratio:.1f}x"
            )

        lines.append("")
        lines.append("=" * 90)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exit Gate (HOLD/CLOSE) Training & Evaluation — Ticket 6"
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
        "--seeds", type=int, nargs="+", default=[42],
        help="Seeds for multi-seed training",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000,
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

    args = parser.parse_args()

    print("=" * 70)
    print("EXIT GATE (HOLD/CLOSE) TRAINING & EVALUATION — Ticket 6")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load data and compute splits
    # ------------------------------------------------------------------
    print("\n[1/4] Loading data and computing splits...")

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

    output_dir = Path(args.output) / "exit_gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: Train with each seed
    # ------------------------------------------------------------------
    print(f"\n[2/4] Training ExitGateEnv (2 actions) with {len(args.seeds)} seeds: {args.seeds}")
    print(f"  Timesteps per seed: {args.timesteps}")

    best_models = {}
    for seed in args.seeds:
        print(f"\n--- Training seed {seed} ---")
        model_path = train_exit_gate_seed(
            seed=seed,
            config_kwargs=config_kwargs,
            train_split=splits["train"],
            val_split=splits["val"],
            total_timesteps=args.timesteps,
            output_dir=output_dir,
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
    # Step 3: Evaluate on test set
    # ------------------------------------------------------------------
    print(f"\n[3/4] Evaluating on test set (bars[{splits['test'][0]}:{splits['test'][1]}])...")

    seed_results = []
    for seed, model_path in best_models.items():
        print(f"\n  Evaluating seed {seed}...")
        result = evaluate_on_test_exit_gate(
            model_path=model_path,
            seed=seed,
            config_kwargs=config_kwargs,
            test_split=splits["test"],
        )
        seed_results.append(result)

        # Print summary
        rl = result.get("rl_exit", {})
        if "error" not in rl:
            print(f"    RL Exit Return: {rl.get('total_return', 0)*100:.2f}% | "
                  f"Trades: {rl.get('num_trades', 0):.0f} | "
                  f"MaxDD: {rl.get('max_drawdown', 0)*100:.2f}%")
        else:
            print(f"    RL Exit: ERROR - {rl.get('error', 'unknown')}")

        bl = result.get("baseline", {})
        if "error" not in bl:
            print(f"    Baseline Return: {bl.get('total_return', 0)*100:.2f}% | "
                  f"Trades: {bl.get('num_trades', 0):.0f} | "
                  f"MaxDD: {bl.get('max_drawdown', 0)*100:.2f}%")

    # ------------------------------------------------------------------
    # Step 4: Generate report and save
    # ------------------------------------------------------------------
    print("\n[4/4] Generating report...")

    report = generate_exit_gate_report(
        seed_results=seed_results,
        splits=splits,
        config={"instrument_id": args.instrument_id, "venue": args.venue},
    )
    print(f"\n{report}")

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.output) / f"exit_gate_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "ticket": "Ticket 6 — RL Exit Gate",
        "config": {
            "instrument_id": args.instrument_id,
            "venue": args.venue,
            "catalog_path": args.catalog_path,
            "seeds": args.seeds,
            "timesteps": args.timesteps,
            "max_steps": args.max_steps,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "splits": splits,
        "seed_results": seed_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
