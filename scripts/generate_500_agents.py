#!/usr/bin/env python3
"""
Generate 500 Agent Configurations

Creates individual agent configs from agents_500_pro.yaml template.
Generates combinations of: symbols x timeframes x algorithms x seeds

Usage:
    python scripts/generate_500_agents.py
    python scripts/generate_500_agents.py --output configs/agents_generated/
    python scripts/generate_500_agents.py --limit 100  # Generate only first 100
"""

import argparse
import json
import yaml
from pathlib import Path
from itertools import product
import hashlib


def load_base_config(config_path: str) -> dict:
    """Load base configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_agent_id(symbol: str, timeframe: str, algorithm: str, seed: int) -> str:
    """Generate unique agent ID."""
    base = f"{symbol}_{timeframe}_{algorithm}_s{seed}"
    # Create short hash for uniqueness
    hash_suffix = hashlib.md5(base.encode()).hexdigest()[:4]
    return f"agent_{symbol.lower()}_{timeframe}_{hash_suffix}"


def generate_configs(
    base_config: dict,
    output_dir: Path,
    limit: int = None,
    seeds_per_combo: int = 3,
) -> list:
    """Generate individual agent configurations."""

    defaults = base_config.get("defaults", {})
    symbols_groups = base_config.get("symbols", {})
    timeframes = base_config.get("timeframes", ["1h"])
    algorithms = base_config.get("algorithms", {"PPO": {}})

    # Flatten symbols
    all_symbols = []
    for group_name, symbols in symbols_groups.items():
        for s in symbols:
            all_symbols.append({
                "symbol": s["symbol"],
                "venue": s["venue"],
                "asset_type": s.get("asset_type", "equity"),
                "group": group_name,
            })

    # Generate combinations
    configs = []
    seeds = list(range(42, 42 + seeds_per_combo))

    for symbol_info, timeframe, (algo_name, algo_params), seed in product(
        all_symbols, timeframes, algorithms.items(), seeds
    ):
        agent_id = generate_agent_id(
            symbol_info["symbol"], timeframe, algo_name, seed
        )

        # Merge configs
        config = {
            "agent_id": agent_id,
            "symbol": symbol_info["symbol"],
            "venue": symbol_info["venue"],
            "asset_type": symbol_info["asset_type"],
            "group": symbol_info["group"],
            "timeframe": timeframe,
            "algorithm": algo_name,
            "seed": seed,

            # Training params
            "total_timesteps": defaults.get("total_timesteps", 5_000_000),
            "learning_rate": algo_params.get("learning_rate", defaults.get("learning_rate", 0.0003)),
            "batch_size": algo_params.get("batch_size", defaults.get("batch_size", 256)),
            "n_epochs": algo_params.get("n_epochs", defaults.get("n_epochs", 10)),
            "gamma": algo_params.get("gamma", defaults.get("gamma", 0.99)),
            "gae_lambda": algo_params.get("gae_lambda", defaults.get("gae_lambda", 0.95)),
            "clip_range": algo_params.get("clip_range", defaults.get("clip_range", 0.2)),
            "ent_coef": algo_params.get("ent_coef", defaults.get("ent_coef", 0.01)),

            # Environment params
            "lookback_period": defaults.get("lookback_period", 20),
            "max_episode_steps": defaults.get("max_episode_steps", 1512),
            "initial_capital": defaults.get("initial_capital", 100000.0),
            "reward_type": defaults.get("reward_type", "sharpe"),

            # Infrastructure
            "n_envs": defaults.get("n_envs", 2),
        }

        configs.append(config)

        if limit and len(configs) >= limit:
            break

        if limit and len(configs) >= limit:
            break

    # Save individual configs
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        config_file = output_dir / f"{config['agent_id']}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    # Save manifest
    manifest = {
        "total_agents": len(configs),
        "symbols": len(all_symbols),
        "timeframes": timeframes,
        "algorithms": list(algorithms.keys()),
        "seeds_per_combo": seeds_per_combo,
        "agents": [c["agent_id"] for c in configs],
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return configs


def generate_batches(configs: list, batch_size: int = 8) -> list:
    """Split configs into batches for parallel training."""
    batches = []

    for i in range(0, len(configs), batch_size):
        batch = configs[i:i + batch_size]
        batches.append({
            "batch_id": f"batch_{i // batch_size:03d}",
            "agents": [c["agent_id"] for c in batch],
            "size": len(batch),
        })

    return batches


def main():
    parser = argparse.ArgumentParser(description="Generate 500 agent configurations")

    parser.add_argument(
        "--config",
        default="configs/agents_500_pro.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--output",
        default="configs/agents_generated",
        help="Output directory for configs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of agents to generate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per symbol/timeframe/algo combo",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Agents per training batch",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GENERATING AGENT CONFIGURATIONS")
    print("=" * 60)

    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    base_config = load_base_config(config_path)
    print(f"  Base config: {config_path}")

    # Generate configs
    output_dir = Path(args.output)
    configs = generate_configs(
        base_config,
        output_dir,
        limit=args.limit,
        seeds_per_combo=args.seeds,
    )

    print(f"  Generated: {len(configs)} agent configs")
    print(f"  Output: {output_dir}")

    # Generate batches
    batches = generate_batches(configs, args.batch_size)

    batches_file = output_dir / "batches.json"
    with open(batches_file, "w") as f:
        json.dump(batches, f, indent=2)

    print(f"  Batches: {len(batches)} (size {args.batch_size})")
    print(f"  Batches file: {batches_file}")

    # Summary by group
    print("\n  By group:")
    groups = {}
    for c in configs:
        g = c["group"]
        groups[g] = groups.get(g, 0) + 1

    for group, count in sorted(groups.items()):
        print(f"    {group}: {count}")

    print("\n  By algorithm:")
    algos = {}
    for c in configs:
        a = c["algorithm"]
        algos[a] = algos.get(a, 0) + 1

    for algo, count in sorted(algos.items()):
        print(f"    {algo}: {count}")

    # Estimate training time
    hours_per_agent = 1.5  # Approximate
    total_hours = len(configs) * hours_per_agent
    parallel_hours = total_hours / args.batch_size

    print(f"\n  Estimated training time:")
    print(f"    Sequential: {total_hours:.0f} hours ({total_hours/24:.1f} days)")
    print(f"    Parallel ({args.batch_size} agents): {parallel_hours:.0f} hours ({parallel_hours/24:.1f} days)")

    # Cost estimate
    gpu_cost_per_hour = 0.17  # RTX A4000
    total_cost = parallel_hours * gpu_cost_per_hour
    print(f"    Estimated cost: ${total_cost:.2f} (at ${gpu_cost_per_hour}/hr)")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
