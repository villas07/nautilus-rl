#!/usr/bin/env python3
"""
RunPod Training Launcher

Launches training jobs on RunPod GPU instances for 500 agents.
Based on GPU_INFRASTRUCTURE_EVAL.md recommendations.

Recommended: RunPod A100 80GB
- 8 agents in parallel per batch
- 44 hours total for 500 agents
- ~$88 total cost

Usage:
    # Estimate cost
    python runpod_launcher.py --estimate --agents 500

    # Launch single batch (8 agents)
    python runpod_launcher.py --batch 0 --gpu-type A100

    # Launch all 500 agents (sequential batches)
    python runpod_launcher.py --full-run --agents 500

    # Resume from specific batch
    python runpod_launcher.py --resume --from-batch 32
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import argparse
import json
import time
import subprocess

import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# RunPod SDK (if available)
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    logger.warning("RunPod SDK not installed. Install with: pip install runpod")

# Max hours a job can stay "RUNNING" before being marked stale
STALE_JOB_TTL_HOURS = 4


@dataclass
class RunPodConfig:
    """Configuration for RunPod training."""

    # GPU Settings - RTX 4090 selected by governance (best price/performance + High availability)
    gpu_type: str = "RTX4090"
    gpu_type_id: str = "NVIDIA GeForce RTX 4090"

    # Pod Resources
    volume_gb: int = 100  # Data catalog + models
    container_disk_gb: int = 50
    min_vcpu: int = 6
    min_memory_gb: int = 31

    # Training Settings
    agents_per_batch: int = 8  # RTX 4090 24GB can handle 8 parallel PPO agents
    timesteps_per_agent: int = 5_000_000
    parallel_envs: int = 4  # Per agent

    # Cost Controls
    max_cost_per_hour: float = 1.00
    max_total_cost: float = 50.0
    auto_shutdown_hours: float = 48  # Safety limit

    # Docker Image - use RunPod's image for better startup command support
    docker_image: str = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"

    # Paths
    workspace: str = "/workspace"
    data_volume: str = "/data"

    # Download URLs (Hetzner VPS - updated 2026-02-02 with 633 instruments)
    data_url: str = "http://46.225.11.110:8080/nautilus/catalog_nautilus_633_20260202.tar.gz"
    code_url: str = "http://46.225.11.110:8080/nautilus/nautilus_code_v2.tar.gz"

    # Hourly costs by GPU type (from RunPod 2026-02-02)
    GPU_COSTS = {
        "RTX4090": 0.59,      # High availability - RECOMMENDED
        "RTX5090": 0.89,      # High availability - Alternative
        "A40": 0.40,          # Low availability
        "L4": 0.39,           # Low availability
        "RTX6000Ada": 0.77,   # Low availability
        "L40S": 0.86,         # Low availability
        "H100": 2.69,         # High availability - Overkill for RL
        "H200": 3.59,         # High availability - Overkill for RL
    }

    # Training speeds (steps/second) - measured with 8 parallel PPO agents
    GPU_SPEEDS = {
        "RTX4090": 4000,      # Ada Lovelace - excellent for RL
        "RTX5090": 5000,      # Blackwell - newest
        "A40": 2500,          # Ampere - older
        "L4": 3000,           # Ada - good efficiency
        "RTX6000Ada": 3500,   # Ada - workstation
        "L40S": 3500,         # Ada - datacenter
        "H100": 5000,         # Hopper - overkill
        "H200": 6000,         # Hopper - overkill
    }


class RunPodLauncher:
    """
    Launches and manages training jobs on RunPod.

    Optimized for 500 agents based on GPU_INFRASTRUCTURE_EVAL.md:
    - A100 80GB: 8 agents parallel, 63 batches, 44 hours, $88
    - Auto-shutdown on completion
    - Checkpoint recovery
    - MLflow tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RunPodConfig] = None,
    ):
        """
        Initialize launcher.

        Args:
            api_key: RunPod API key (or RUNPOD_API_KEY env var).
            config: RunPod configuration.
        """
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.config = config or RunPodConfig()

        if RUNPOD_AVAILABLE and self.api_key:
            runpod.api_key = self.api_key

        self.active_pods: List[str] = []
        self.job_tracker: Dict[str, Any] = {}

        # State file for recovery
        self.state_file = Path(__file__).parent.parent / "training_state.json"

    def estimate_cost(
        self,
        n_agents: int,
        timesteps: Optional[int] = None,
        gpu_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate training cost for given number of agents.

        Args:
            n_agents: Number of agents to train.
            timesteps: Timesteps per agent (default from config).
            gpu_type: GPU type (default from config).

        Returns:
            Detailed cost breakdown.
        """
        timesteps = timesteps or self.config.timesteps_per_agent
        gpu_type = gpu_type or self.config.gpu_type

        hourly_cost = self.config.GPU_COSTS.get(gpu_type, 2.0)
        steps_per_sec = self.config.GPU_SPEEDS.get(gpu_type, 2000)

        # Time per agent (in hours)
        seconds_per_agent = timesteps / steps_per_sec
        hours_per_agent = seconds_per_agent / 3600

        # With parallelization
        parallel = self.config.agents_per_batch
        effective_hours = (n_agents / parallel) * hours_per_agent

        # Batches
        n_batches = (n_agents + parallel - 1) // parallel

        # Total cost
        total_cost = effective_hours * hourly_cost

        return {
            "n_agents": n_agents,
            "timesteps_per_agent": timesteps,
            "gpu_type": gpu_type,
            "hourly_cost_usd": hourly_cost,
            "agents_per_batch": parallel,
            "n_batches": n_batches,
            "hours_per_batch": hours_per_agent,
            "total_hours": effective_hours,
            "total_cost_usd": total_cost,
            "cost_per_agent_usd": total_cost / n_agents,
            "estimated_completion": f"{effective_hours:.1f} hours ({effective_hours/24:.1f} days)",
        }

    def create_training_script(
        self,
        batch_id: int,
        agent_ids: List[str],
    ) -> str:
        """
        Create the training script for a batch.

        Args:
            batch_id: Batch number.
            agent_ids: List of agent IDs to train.

        Returns:
            Shell script content.
        """
        agents_str = " ".join(agent_ids)
        n_agents = len(agent_ids)

        return f'''#!/bin/bash
set -e

echo "=========================================="
echo "NAUTILUS AGENTS - BATCH {batch_id}"
echo "Agents: {agents_str}"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="${{MLFLOW_TRACKING_URI:-http://host.docker.internal:5000}}"

# Install dependencies
pip install --quiet nautilus_trader stable-baselines3 gymnasium mlflow structlog pyyaml

# Download training code if not exists
CODE_DIR="{self.config.workspace}/nautilus-agents"
if [ ! -d "$CODE_DIR" ]; then
    echo "Downloading training code from Hetzner VPS..."
    cd {self.config.workspace}
    curl -L -o nautilus-agents-code.tar.gz {self.config.code_url}
    echo "Extracting code..."
    mkdir -p nautilus-agents
    tar -xzf nautilus-agents-code.tar.gz -C nautilus-agents
    rm nautilus-agents-code.tar.gz
    echo "Code ready: $(ls -la $CODE_DIR)"
fi

# Download data catalog if not exists
DATA_DIR="{self.config.workspace}/data/catalog"
if [ ! -d "$DATA_DIR" ]; then
    echo "Downloading data catalog from Hetzner VPS..."
    mkdir -p {self.config.workspace}/data
    cd {self.config.workspace}/data
    curl -L -o catalog.tar.gz {self.config.data_url}
    echo "Extracting data..."
    tar -xzf catalog.tar.gz
    rm catalog.tar.gz
    echo "Data catalog ready: $(du -sh $DATA_DIR)"
fi

# Navigate to code directory
cd {self.config.workspace}/nautilus-agents

# Create symlink to data
mkdir -p data
ln -sf {self.config.workspace}/data/catalog data/catalog

# Train each agent in parallel batches
PIDS=()
AGENT_IDS=({agents_str})

for AGENT_ID in "${{AGENT_IDS[@]}}"; do
    echo "Starting training for $AGENT_ID..."

    python training/train_agent.py \\
        --agent-id "$AGENT_ID" \\
        --symbol SPY \\
        --timesteps {self.config.timesteps_per_agent} \\
        --output-dir {self.config.workspace}/models \\
        --catalog-path {self.config.data_volume}/catalog \\
        --log-dir {self.config.workspace}/logs &

    PIDS+=($!)

    # Small delay to prevent race conditions
    sleep 2
done

# Wait for all training processes
echo "Waiting for {n_agents} agents to complete..."
FAILED=0
for PID in "${{PIDS[@]}}"; do
    if ! wait $PID; then
        echo "Process $PID failed"
        FAILED=$((FAILED + 1))
    fi
done

# Copy models to data volume for persistence
echo "Copying models to persistent volume..."
mkdir -p {self.config.data_volume}/models
cp -r {self.config.workspace}/models/* {self.config.data_volume}/models/

# Generate batch report
echo "=========================================="
echo "BATCH {batch_id} COMPLETE"
echo "Finished: $(date)"
echo "Agents trained: {n_agents}"
echo "Failed: $FAILED"
echo "=========================================="

# Write completion marker
echo '{{"batch_id": {batch_id}, "agents": {json.dumps(agent_ids)}, "completed": true, "failed": '$FAILED'}}' > {self.config.data_volume}/batch_{batch_id}_complete.json

# Auto-shutdown if all batches complete
if [ "$RUNPOD_POD_ID" != "" ]; then
    echo "Signaling pod completion..."
    # Pod will auto-terminate based on RunPod settings
fi

exit $FAILED
'''

    def generate_agent_ids(
        self,
        start: int,
        count: int,
        prefix: str = "agent",
    ) -> List[str]:
        """Generate agent IDs for a range."""
        return [f"{prefix}_{str(i).zfill(3)}" for i in range(start, start + count)]

    def get_batch_agents(
        self,
        batch_id: int,
        total_agents: int = 500,
    ) -> List[str]:
        """Get agent IDs for a specific batch."""
        start = batch_id * self.config.agents_per_batch
        count = min(self.config.agents_per_batch, total_agents - start)

        if count <= 0:
            return []

        return self.generate_agent_ids(start, count)

    def launch_pod(
        self,
        batch_id: int,
        agent_ids: List[str],
        data_volume_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Launch a RunPod instance for training.

        Args:
            batch_id: Batch identifier.
            agent_ids: Agents to train in this batch.
            data_volume_id: Existing data volume ID.

        Returns:
            Pod ID if successful.
        """
        if not RUNPOD_AVAILABLE:
            logger.error("RunPod SDK not available")
            return None

        if not self.api_key:
            logger.error("RUNPOD_API_KEY not set")
            return None

        # GPU type mapping (updated 2026-02-02)
        gpu_map = {
            "RTX4090": "NVIDIA GeForce RTX 4090",    # RECOMMENDED
            "RTX5090": "NVIDIA GeForce RTX 5090",    # Alternative
            "A40": "NVIDIA A40",
            "L4": "NVIDIA L4",
            "RTX6000Ada": "NVIDIA RTX 6000 Ada",
            "L40S": "NVIDIA L40S",
            "H100": "NVIDIA H100 80GB HBM3",
            "H200": "NVIDIA H200 SXM",
        }

        # Startup command: download and run the training script
        startup_script_url = "http://46.225.11.110:8080/nautilus/pod_startup.sh"

        # Agent IDs as space-separated string for the startup script
        agent_ids_str = " ".join(agent_ids)

        pod_config = {
            "name": f"nautilus-batch-{batch_id}",
            "image_name": self.config.docker_image,
            "gpu_type_id": gpu_map.get(self.config.gpu_type, self.config.gpu_type_id),
            "cloud_type": "SECURE",
            "volume_in_gb": self.config.volume_gb,
            "container_disk_in_gb": self.config.container_disk_gb,
            "min_vcpu_count": self.config.min_vcpu,
            "min_memory_in_gb": self.config.min_memory_gb,
            "gpu_count": 1,
            "ports": "8888/http,22/tcp",
            "volume_mount_path": self.config.data_volume,
            "start_ssh": True,  # Enable SSH for debugging
            "env": {
                "BATCH_ID": str(batch_id),
                "AGENT_IDS": agent_ids_str,
                "DATA_URL": self.config.data_url,
                "CODE_URL": self.config.code_url,
                "STARTUP_SCRIPT": startup_script_url,
                "KEEP_ALIVE": "true",  # Keep alive for monitoring
                "PUBLIC_KEY": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDQQWcq+RayJZDpS2nYlO7jtMi8WKL2h+kMbbUjkTE07",
            },
        }

        # If existing volume, attach it
        if data_volume_id:
            pod_config["volume_id"] = data_volume_id

        try:
            pod = runpod.create_pod(**pod_config)
            pod_id = pod["id"]

            logger.info(
                "Pod launched",
                pod_id=pod_id,
                batch_id=batch_id,
                agents=len(agent_ids),
                gpu=self.config.gpu_type,
            )

            # Track active pod
            self.active_pods.append(pod_id)
            self.job_tracker[pod_id] = {
                "batch_id": batch_id,
                "agents": agent_ids,
                "status": "RUNNING",
                "started": datetime.now().isoformat(),
            }

            # Save state
            self._save_state()

            return pod_id

        except Exception as e:
            logger.error(f"Failed to launch pod: {e}")
            return None

    def execute_training_on_pod(self, pod_id: str) -> bool:
        """
        Execute the training script on a running pod.

        This is called after the pod is ready (has SSH access).

        Args:
            pod_id: The pod ID to execute on.

        Returns:
            Success status.
        """
        if not RUNPOD_AVAILABLE:
            logger.error("RunPod SDK not available")
            return False

        try:
            # Get pod details for SSH connection
            pod = runpod.get_pod(pod_id)

            if not pod:
                logger.error(f"Pod {pod_id} not found")
                return False

            status = pod.get("desiredStatus")
            if status != "RUNNING":
                logger.error(f"Pod {pod_id} is not running (status: {status})")
                return False

            # Get SSH connection info
            runtime = pod.get("runtime", {})
            if not runtime:
                logger.error(f"Pod {pod_id} runtime not ready yet")
                return False

            # Use runpod's execute endpoint if available
            # Otherwise provide SSH command
            startup_url = pod.get("env", {}).get("STARTUP_SCRIPT",
                "http://46.225.11.110:8080/nautilus/pod_startup.sh")

            cmd = f"curl -sL {startup_url} | bash"

            logger.info(f"Executing training on pod {pod_id}...")
            logger.info(f"Command: {cmd}")

            # Try to execute via runpod API
            try:
                result = runpod.run_pod_command(pod_id, cmd)
                logger.info(f"Execution started: {result}")
                return True
            except AttributeError:
                # run_pod_command not available, provide manual instructions
                ssh_cmd = f"ssh root@{pod_id}.runpod.io \"{cmd}\""
                logger.info(f"Manual execution required:")
                logger.info(f"  {ssh_cmd}")
                print(f"\nTo start training manually, run:")
                print(f"  {ssh_cmd}")
                print(f"\nOr connect via RunPod web terminal and run:")
                print(f"  {cmd}")
                return True

        except Exception as e:
            logger.error(f"Failed to execute on pod: {e}")
            return False

    def wait_for_pod_ready(
        self,
        pod_id: str,
        timeout: int = 300,
        interval: int = 10,
    ) -> bool:
        """
        Wait for a pod to be ready (SSH accessible).

        Args:
            pod_id: Pod ID to wait for.
            timeout: Max wait time in seconds.
            interval: Check interval in seconds.

        Returns:
            True if pod is ready, False if timeout.
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pod = runpod.get_pod(pod_id)
                runtime = pod.get("runtime", {}) or {}
                uptime = runtime.get("uptimeInSeconds", 0)

                if uptime > 0:
                    logger.info(f"Pod {pod_id} is ready (uptime: {uptime}s)")
                    return True

                status = pod.get("desiredStatus", "UNKNOWN")
                logger.info(f"Waiting for pod {pod_id}... ({status}, uptime: {uptime}s)")

            except Exception as e:
                logger.warning(f"Error checking pod: {e}")

            time.sleep(interval)

        logger.error(f"Timeout waiting for pod {pod_id}")
        return False

    def upload_data_to_volume(
        self,
        local_catalog_path: str,
        volume_id: str,
    ) -> bool:
        """
        Upload data catalog to RunPod volume.

        Args:
            local_catalog_path: Local path to data catalog.
            volume_id: RunPod volume ID.

        Returns:
            Success status.
        """
        logger.info("Uploading data catalog to RunPod volume...")

        # This would use runpodctl or rsync over SSH
        # For now, provide instructions
        print(f"""
To upload data catalog to RunPod:

1. Create a network volume in RunPod dashboard
2. Start a temporary pod with the volume attached
3. Use rsync or scp to upload:

   rsync -avz --progress {local_catalog_path}/ \\
       root@<pod-ip>:{self.config.data_volume}/catalog/

4. Also upload the nautilus-agents code:

   rsync -avz --progress /path/to/nautilus-agents/ \\
       root@<pod-ip>:{self.config.workspace}/nautilus-agents/

5. Terminate the temporary pod (volume persists)
""")
        return True

    def monitor_pods(self) -> Dict[str, Any]:
        """Monitor all active pods."""
        if not RUNPOD_AVAILABLE:
            return {"error": "RunPod SDK not available"}

        results = {}
        for pod_id in self.active_pods:
            try:
                pod = runpod.get_pod(pod_id)
                results[pod_id] = {
                    "batch_id": self.job_tracker.get(pod_id, {}).get("batch_id"),
                    "status": pod.get("desiredStatus"),
                    "runtime_hours": pod.get("runtime", {}).get("uptimeInSeconds", 0) / 3600,
                    "cost_so_far": pod.get("costPerHr", 0) * pod.get("runtime", {}).get("uptimeInSeconds", 0) / 3600,
                }
            except Exception as e:
                results[pod_id] = {"error": str(e)}

        return results

    def run_full_training(
        self,
        n_agents: int = 500,
        start_batch: int = 0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete training for all agents.

        Args:
            n_agents: Total number of agents to train.
            start_batch: Starting batch (for resume).
            dry_run: If True, only print what would be done.

        Returns:
            Training summary.
        """
        n_batches = (n_agents + self.config.agents_per_batch - 1) // self.config.agents_per_batch
        estimate = self.estimate_cost(n_agents)

        print("\n" + "=" * 60)
        print("NAUTILUS AGENTS - FULL TRAINING RUN")
        print("=" * 60)
        print(f"Total agents: {n_agents}")
        print(f"Batches: {n_batches}")
        print(f"Agents per batch: {self.config.agents_per_batch}")
        print(f"GPU type: {self.config.gpu_type}")
        print(f"Estimated time: {estimate['estimated_completion']}")
        print(f"Estimated cost: ${estimate['total_cost_usd']:.2f}")
        print("=" * 60)

        if dry_run:
            print("\n[DRY RUN] Would launch the following batches:")
            for batch_id in range(start_batch, n_batches):
                agents = self.get_batch_agents(batch_id, n_agents)
                print(f"  Batch {batch_id}: {agents}")
            return {"dry_run": True, "batches": n_batches}

        # Launch batches sequentially (one pod at a time for cost control)
        results = {
            "total_agents": n_agents,
            "batches_launched": 0,
            "batches_completed": 0,
            "pods": [],
            "start_time": datetime.now().isoformat(),
        }

        for batch_id in range(start_batch, n_batches):
            agents = self.get_batch_agents(batch_id, n_agents)
            if not agents:
                continue

            print(f"\nLaunching batch {batch_id + 1}/{n_batches}...")
            pod_id = self.launch_pod(batch_id, agents)

            if pod_id:
                results["batches_launched"] += 1
                results["pods"].append({
                    "pod_id": pod_id,
                    "batch_id": batch_id,
                    "agents": agents,
                })

                # Wait for pod to be ready (SSH accessible)
                print(f"Pod {pod_id} launched. Waiting for it to be ready...")
                if self.wait_for_pod_ready(pod_id, timeout=300):
                    # Execute training script
                    print(f"Pod ready. Starting training...")
                    self.execute_training_on_pod(pod_id)

                    # Wait for training to complete
                    print(f"Training started. Waiting for completion...")
                    self._wait_for_pod(pod_id)
                    results["batches_completed"] += 1
                else:
                    print(f"Pod {pod_id} failed to become ready. Terminating...")
                    runpod.terminate_pod(pod_id)
            else:
                print(f"Failed to launch batch {batch_id}")
                break

        results["end_time"] = datetime.now().isoformat()

        # Save final state
        self._save_state()

        return results

    def _wait_for_pod(
        self,
        pod_id: str,
        poll_interval: int = 60,
        timeout: int = 3600 * 6,  # 6 hours max per batch
    ) -> bool:
        """Wait for a pod to complete."""
        if not RUNPOD_AVAILABLE:
            # Simulate for testing
            time.sleep(5)
            return True

        start = time.time()
        while time.time() - start < timeout:
            try:
                pod = runpod.get_pod(pod_id)
                status = pod.get("desiredStatus")

                if status in ["TERMINATED", "EXITED"]:
                    logger.info(f"Pod {pod_id} completed")
                    return True

                logger.info(f"Pod {pod_id} status: {status}")

            except Exception as e:
                logger.warning(f"Error checking pod {pod_id}: {e}")

            time.sleep(poll_interval)

        logger.error(f"Pod {pod_id} timed out after {timeout/3600:.1f} hours")
        return False

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a specific pod."""
        if not RUNPOD_AVAILABLE:
            return False

        try:
            runpod.terminate_pod(pod_id)
            logger.info(f"Terminated pod {pod_id}")
            if pod_id in self.active_pods:
                self.active_pods.remove(pod_id)
            return True
        except Exception as e:
            logger.error(f"Failed to terminate {pod_id}: {e}")
            return False

    def terminate_all(self) -> None:
        """Emergency stop - terminate all active pods."""
        for pod_id in list(self.active_pods):
            self.terminate_pod(pod_id)
        self.active_pods = []
        self._save_state()

    def _save_state(self) -> None:
        """Save launcher state for recovery."""
        state = {
            "active_pods": self.active_pods,
            "job_tracker": self.job_tracker,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load previous state for recovery."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.active_pods = state.get("active_pods", [])
                self.job_tracker = state.get("job_tracker", {})
            self.check_stale_jobs()

    def check_stale_jobs(self) -> None:
        """Detect and mark RUNNING jobs that exceeded TTL."""
        now = datetime.utcnow()
        ttl = timedelta(hours=STALE_JOB_TTL_HOURS)

        for job in self.job_tracker.values():
            if job.get("status") == "RUNNING" and "started" in job:
                try:
                    started = datetime.fromisoformat(job["started"])
                    if now - started > ttl:
                        job["status"] = "STALE"
                except Exception:
                    pass


def print_estimate(estimate: Dict[str, Any]) -> None:
    """Pretty print cost estimate."""
    print("\n" + "=" * 50)
    print("COST ESTIMATE")
    print("=" * 50)
    for key, value in estimate.items():
        if isinstance(value, float):
            if "cost" in key or "usd" in key.lower():
                print(f"  {key}: ${value:.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Launch RunPod training for Nautilus Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate cost for 500 agents
  python runpod_launcher.py --estimate --agents 500

  # Launch a single batch
  python runpod_launcher.py --batch 0

  # Launch full training run
  python runpod_launcher.py --full-run --agents 500

  # Resume from batch 32
  python runpod_launcher.py --full-run --agents 500 --from-batch 32

  # Monitor active pods
  python runpod_launcher.py --monitor

  # Emergency stop all pods
  python runpod_launcher.py --terminate-all
        """,
    )

    parser.add_argument(
        "--agents", type=int, default=500,
        help="Total number of agents to train (default: 500)",
    )
    parser.add_argument(
        "--gpu-type", type=str, default="RTX4090",
        choices=["RTX4090", "RTX5090", "A40", "L4", "RTX6000Ada", "L40S", "H100", "H200"],
        help="GPU type (default: RTX4090 - best price/performance with High availability)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Timesteps per agent (default: 5M)",
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Only show cost estimate",
    )
    parser.add_argument(
        "--batch", type=int,
        help="Launch a specific batch",
    )
    parser.add_argument(
        "--full-run", action="store_true",
        help="Launch complete training run",
    )
    parser.add_argument(
        "--from-batch", type=int, default=0,
        help="Start from specific batch (for resume)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without launching",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Monitor active pods",
    )
    parser.add_argument(
        "--terminate-all", action="store_true",
        help="Emergency stop all pods",
    )
    parser.add_argument(
        "--upload-data", type=str,
        help="Upload data catalog to volume",
    )

    args = parser.parse_args()

    # Create launcher with config
    config = RunPodConfig(
        gpu_type=args.gpu_type,
        timesteps_per_agent=args.timesteps,
    )
    launcher = RunPodLauncher(config=config)

    # Load previous state
    launcher._load_state()

    # Handle commands
    if args.estimate:
        estimate = launcher.estimate_cost(args.agents)
        print_estimate(estimate)
        return

    if args.monitor:
        results = launcher.monitor_pods()
        print("\nActive Pods:")
        for pod_id, status in results.items():
            print(f"  {pod_id}: {status}")
        return

    if args.terminate_all:
        print("Terminating all active pods...")
        launcher.terminate_all()
        print("Done.")
        return

    if args.upload_data:
        launcher.upload_data_to_volume(args.upload_data, "VOLUME_ID")
        return

    if args.batch is not None:
        agents = launcher.get_batch_agents(args.batch, args.agents)
        if args.dry_run:
            print(f"Would launch batch {args.batch} with agents: {agents}")
        else:
            pod_id = launcher.launch_pod(args.batch, agents)
            if pod_id:
                print(f"Launched pod {pod_id} for batch {args.batch}")
                print("Waiting for pod to be ready...")
                if launcher.wait_for_pod_ready(pod_id, timeout=300):
                    print("Pod ready! Starting training...")
                    launcher.execute_training_on_pod(pod_id)
                    print(f"\nPod {pod_id} is training. Monitor with:")
                    print(f"  python training/runpod_launcher.py --monitor")
                else:
                    print("Pod failed to become ready. Check RunPod dashboard.")
        return

    if args.full_run:
        results = launcher.run_full_training(
            n_agents=args.agents,
            start_batch=args.from_batch,
            dry_run=args.dry_run,
        )
        print("\nTraining Summary:")
        print(json.dumps(results, indent=2))
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
