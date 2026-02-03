#!/usr/bin/env python3
"""
Launch Batch Training on RunPod

Orchestrates training of multiple agent batches on RunPod.
Handles pod lifecycle, uploads, training, and model download.

Usage:
    python scripts/launch_batch_training.py --batch batch_000
    python scripts/launch_batch_training.py --batches batch_000 batch_001 batch_002
    python scripts/launch_batch_training.py --all  # Train all batches
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime


def load_batches(batches_file: str) -> list:
    """Load batch definitions."""
    with open(batches_file) as f:
        return json.load(f)


def load_agent_config(config_dir: Path, agent_id: str) -> dict:
    """Load individual agent config."""
    config_file = config_dir / f"{agent_id}.json"
    with open(config_file) as f:
        return json.load(f)


def run_ssh_command(host: str, port: int, command: str) -> tuple:
    """Run SSH command and return output."""
    ssh_cmd = f'ssh -p {port} {host} "{command}"'
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def wait_for_training(host: str, port: int, check_interval: int = 300) -> bool:
    """Wait for training to complete."""
    print(f"  Waiting for training to complete (checking every {check_interval}s)...")

    while True:
        code, out, _ = run_ssh_command(
            host, port,
            "ps aux | grep train_agent | grep -v grep | wc -l"
        )

        if code != 0:
            print("  WARNING: SSH connection failed, retrying...")
            time.sleep(60)
            continue

        running = int(out.strip())

        if running == 0:
            print("  Training complete!")
            return True

        print(f"  {running} agents still training...")
        time.sleep(check_interval)


def download_models(host: str, port: int, local_dir: str) -> bool:
    """Download trained models."""
    cmd = f"scp -P {port} -r {host}:/workspace/models/* {local_dir}/"
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def train_batch(
    batch: dict,
    config_dir: Path,
    host: str,
    port: int,
    catalog_path: str,
    output_dir: str,
    timesteps: int,
) -> dict:
    """Train a single batch of agents."""
    batch_id = batch["batch_id"]
    agents = batch["agents"]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING BATCH: {batch_id}")
    print(f"  Agents: {len(agents)}")
    print(f"{'=' * 60}")

    results = {
        "batch_id": batch_id,
        "agents": agents,
        "start_time": datetime.now().isoformat(),
        "status": "starting",
    }

    # Start training for each agent
    for agent_id in agents:
        config = load_agent_config(config_dir, agent_id)

        symbol = config["symbol"]
        venue = config["venue"]

        cmd = (
            f"nohup python -u training/train_agent.py "
            f"--agent-id {agent_id} "
            f"--symbol {symbol} "
            f"--venue {venue} "
            f"--timesteps {timesteps} "
            f"--output-dir {output_dir} "
            f"--catalog-path {catalog_path} "
            f"--log-dir /workspace/logs "
            f"> /workspace/logs/{agent_id}.log 2>&1 &"
        )

        code, _, err = run_ssh_command(host, port, cmd)

        if code == 0:
            print(f"  Started: {agent_id} ({symbol}.{venue})")
        else:
            print(f"  FAILED: {agent_id} - {err}")
            results["status"] = "partial_failure"

    results["status"] = "training"
    return results


def main():
    parser = argparse.ArgumentParser(description="Launch batch training on RunPod")

    parser.add_argument(
        "--batch",
        help="Single batch to train",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        help="Multiple batches to train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all batches",
    )
    parser.add_argument(
        "--batches-file",
        default="configs/agents_generated/batches.json",
        help="Batches definition file",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/agents_generated",
        help="Directory with agent configs",
    )
    parser.add_argument(
        "--host",
        default="root@193.183.22.54",
        help="RunPod SSH host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1795,
        help="RunPod SSH port",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Training timesteps per agent",
    )
    parser.add_argument(
        "--catalog-path",
        default="/workspace/data/catalog",
        help="Path to data catalog on RunPod",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/models",
        help="Output directory on RunPod",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for training to complete",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download models after training",
    )
    parser.add_argument(
        "--local-models-dir",
        default="./models",
        help="Local directory for downloaded models",
    )

    args = parser.parse_args()

    # Load batches
    batches_file = Path(args.batches_file)
    if not batches_file.exists():
        print(f"ERROR: Batches file not found: {batches_file}")
        print("Run: python scripts/generate_500_agents.py first")
        return 1

    all_batches = load_batches(batches_file)
    config_dir = Path(args.config_dir)

    # Select batches to train
    if args.all:
        batches_to_train = all_batches
    elif args.batches:
        batches_to_train = [b for b in all_batches if b["batch_id"] in args.batches]
    elif args.batch:
        batches_to_train = [b for b in all_batches if b["batch_id"] == args.batch]
    else:
        print("ERROR: Specify --batch, --batches, or --all")
        return 1

    if not batches_to_train:
        print("ERROR: No matching batches found")
        return 1

    print(f"Training {len(batches_to_train)} batches")
    print(f"Total agents: {sum(b['size'] for b in batches_to_train)}")

    # Train each batch
    results = []
    for batch in batches_to_train:
        result = train_batch(
            batch,
            config_dir,
            args.host,
            args.port,
            args.catalog_path,
            args.output_dir,
            args.timesteps,
        )
        results.append(result)

    # Wait for completion if requested
    if args.wait:
        wait_for_training(args.host, args.port)

        # Update results
        for r in results:
            r["status"] = "completed"
            r["end_time"] = datetime.now().isoformat()

    # Download if requested
    if args.download:
        print("\nDownloading models...")
        local_dir = Path(args.local_models_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        if download_models(args.host, args.port, str(local_dir)):
            print(f"  Models downloaded to: {local_dir}")
        else:
            print("  WARNING: Download failed")

    # Save results
    results_file = Path(f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_file}")

    return 0


if __name__ == "__main__":
    exit(main())
