#!/usr/bin/env python3
"""
RunPod Training Launcher

Launches training jobs on RunPod GPU instances.

Usage:
    python runpod_launcher.py --agents 0-99 --gpu-type A100
    python runpod_launcher.py --config configs/runpod_batch.yaml
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import argparse
import json
import time

import structlog

logger = structlog.get_logger()

# RunPod SDK (if available)
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False
    logger.warning("RunPod SDK not installed. Install with: pip install runpod")


# Default RunPod configuration
DEFAULT_POD_CONFIG = {
    "name": "nautilus-training",
    "image_name": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
    "gpu_type_id": "NVIDIA A100-SXM4-80GB",
    "cloud_type": "SECURE",
    "volume_in_gb": 50,
    "container_disk_in_gb": 20,
    "min_vcpu_count": 8,
    "min_memory_in_gb": 32,
    "gpu_count": 1,
    "ports": "8888/http,22/tcp",
    "volume_mount_path": "/workspace",
    "env": {},
}


class RunPodLauncher:
    """
    Launches and manages training jobs on RunPod.

    Features:
    - Automatic pod provisioning
    - Job monitoring
    - Results retrieval
    - Cost tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        pod_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize launcher.

        Args:
            api_key: RunPod API key.
            pod_config: Pod configuration overrides.
        """
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.pod_config = {**DEFAULT_POD_CONFIG, **(pod_config or {})}

        if RUNPOD_AVAILABLE and self.api_key:
            runpod.api_key = self.api_key

        self.active_pods: List[str] = []

    def launch_training_pod(
        self,
        agent_range: str,
        gpu_type: str = "A100",
        max_cost_per_hour: float = 2.0,
    ) -> Optional[str]:
        """
        Launch a training pod on RunPod.

        Args:
            agent_range: Range of agents to train (e.g., "0-99").
            gpu_type: GPU type (A100, A40, RTX3090, etc.).
            max_cost_per_hour: Maximum hourly cost.

        Returns:
            Pod ID if successful.
        """
        if not RUNPOD_AVAILABLE:
            logger.error("RunPod SDK not available")
            return None

        if not self.api_key:
            logger.error("RunPod API key not set")
            return None

        # Map GPU type to RunPod ID
        gpu_map = {
            "A100": "NVIDIA A100-SXM4-80GB",
            "A100-80GB": "NVIDIA A100-SXM4-80GB",
            "A40": "NVIDIA A40",
            "RTX3090": "NVIDIA GeForce RTX 3090",
            "RTX4090": "NVIDIA GeForce RTX 4090",
        }

        gpu_type_id = gpu_map.get(gpu_type, gpu_type)

        # Create startup script
        startup_script = self._create_startup_script(agent_range)

        # Pod configuration
        pod_config = {
            **self.pod_config,
            "gpu_type_id": gpu_type_id,
            "docker_args": f"bash -c '{startup_script}'",
            "env": {
                "AGENT_RANGE": agent_range,
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", ""),
            },
        }

        try:
            # Create pod
            pod = runpod.create_pod(**pod_config)
            pod_id = pod["id"]

            logger.info(
                "Pod created",
                pod_id=pod_id,
                gpu_type=gpu_type,
                agent_range=agent_range,
            )

            self.active_pods.append(pod_id)
            return pod_id

        except Exception as e:
            logger.error(f"Failed to create pod: {e}")
            return None

    def _create_startup_script(self, agent_range: str) -> str:
        """Create the startup script for the pod."""
        return f"""
        # Setup environment
        pip install nautilus_trader stable-baselines3 gymnasium mlflow

        # Clone repository
        git clone https://github.com/your-repo/nautilus-agents.git /workspace/nautilus-agents
        cd /workspace/nautilus-agents

        # Download data catalog
        python training/download_models.py --download-data

        # Run training
        python training/train_batch.py --agents {agent_range} --parallel 1

        # Upload models
        python training/download_models.py --upload-models

        # Shutdown pod after completion
        runpodctl stop pod $RUNPOD_POD_ID
        """

    def monitor_pod(self, pod_id: str) -> Dict[str, Any]:
        """
        Monitor pod status.

        Args:
            pod_id: Pod ID to monitor.

        Returns:
            Pod status information.
        """
        if not RUNPOD_AVAILABLE:
            return {"error": "RunPod SDK not available"}

        try:
            pod = runpod.get_pod(pod_id)
            return {
                "id": pod_id,
                "status": pod.get("desiredStatus"),
                "gpu": pod.get("gpuTypeId"),
                "runtime": pod.get("runtime", {}).get("uptimeInSeconds", 0),
                "cost": pod.get("costPerHr", 0),
            }
        except Exception as e:
            return {"id": pod_id, "error": str(e)}

    def wait_for_completion(
        self,
        pod_ids: List[str],
        poll_interval: int = 60,
        timeout: int = 3600 * 12,  # 12 hours
    ) -> Dict[str, Any]:
        """
        Wait for pods to complete.

        Args:
            pod_ids: List of pod IDs.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds.

        Returns:
            Final status of all pods.
        """
        start_time = time.time()
        results = {}

        while time.time() - start_time < timeout:
            all_done = True

            for pod_id in pod_ids:
                if pod_id in results and results[pod_id].get("status") in ["TERMINATED", "EXITED"]:
                    continue

                status = self.monitor_pod(pod_id)
                results[pod_id] = status

                if status.get("status") not in ["TERMINATED", "EXITED"]:
                    all_done = False

                logger.info(f"Pod {pod_id}: {status.get('status')}")

            if all_done:
                break

            time.sleep(poll_interval)

        return results

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod."""
        if not RUNPOD_AVAILABLE:
            return False

        try:
            runpod.terminate_pod(pod_id)
            logger.info(f"Terminated pod {pod_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return False

    def terminate_all(self) -> None:
        """Terminate all active pods."""
        for pod_id in self.active_pods:
            self.terminate_pod(pod_id)
        self.active_pods = []

    def estimate_cost(
        self,
        n_agents: int,
        timesteps_per_agent: int = 5_000_000,
        gpu_type: str = "A100",
    ) -> Dict[str, float]:
        """
        Estimate training cost.

        Args:
            n_agents: Number of agents to train.
            timesteps_per_agent: Timesteps per agent.
            gpu_type: GPU type.

        Returns:
            Cost estimates.
        """
        # Approximate training speeds (timesteps/hour)
        speeds = {
            "A100": 500_000,
            "A40": 300_000,
            "RTX4090": 400_000,
            "RTX3090": 250_000,
        }

        # Hourly costs (approximate)
        costs = {
            "A100": 1.50,
            "A40": 0.80,
            "RTX4090": 0.60,
            "RTX3090": 0.40,
        }

        speed = speeds.get(gpu_type, 300_000)
        hourly_cost = costs.get(gpu_type, 1.0)

        hours_per_agent = timesteps_per_agent / speed
        total_hours = hours_per_agent * n_agents
        total_cost = total_hours * hourly_cost

        return {
            "n_agents": n_agents,
            "gpu_type": gpu_type,
            "hours_per_agent": hours_per_agent,
            "total_hours": total_hours,
            "hourly_cost": hourly_cost,
            "total_cost": total_cost,
            "cost_per_agent": total_cost / n_agents,
        }


def launch_distributed_training(
    n_agents: int = 500,
    agents_per_pod: int = 50,
    gpu_type: str = "A100",
) -> List[str]:
    """
    Launch distributed training across multiple pods.

    Args:
        n_agents: Total number of agents.
        agents_per_pod: Agents per pod.
        gpu_type: GPU type for all pods.

    Returns:
        List of pod IDs.
    """
    launcher = RunPodLauncher()

    n_pods = (n_agents + agents_per_pod - 1) // agents_per_pod
    pod_ids = []

    for i in range(n_pods):
        start = i * agents_per_pod
        end = min((i + 1) * agents_per_pod - 1, n_agents - 1)
        agent_range = f"{start}-{end}"

        pod_id = launcher.launch_training_pod(
            agent_range=agent_range,
            gpu_type=gpu_type,
        )

        if pod_id:
            pod_ids.append(pod_id)

        # Small delay between launches
        time.sleep(5)

    logger.info(f"Launched {len(pod_ids)} pods for {n_agents} agents")
    return pod_ids


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Launch RunPod training jobs")

    parser.add_argument(
        "--agents",
        type=str,
        default="0-99",
        help="Agent range (e.g., '0-99')",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="A100",
        help="GPU type (A100, A40, RTX3090, RTX4090)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Only estimate costs, don't launch",
    )
    parser.add_argument(
        "--distributed",
        type=int,
        help="Launch distributed training with N total agents",
    )
    parser.add_argument(
        "--agents-per-pod",
        type=int,
        default=50,
        help="Agents per pod for distributed training",
    )

    args = parser.parse_args()

    launcher = RunPodLauncher()

    if args.estimate:
        # Parse agent range to get count
        if "-" in args.agents:
            start, end = args.agents.split("-")
            n_agents = int(end) - int(start) + 1
        else:
            n_agents = 1

        estimate = launcher.estimate_cost(
            n_agents=n_agents,
            gpu_type=args.gpu_type,
        )

        print("\n" + "=" * 50)
        print("COST ESTIMATE")
        print("=" * 50)
        for key, value in estimate.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        return

    if args.distributed:
        pod_ids = launch_distributed_training(
            n_agents=args.distributed,
            agents_per_pod=args.agents_per_pod,
            gpu_type=args.gpu_type,
        )

        print(f"\nLaunched {len(pod_ids)} pods:")
        for pod_id in pod_ids:
            print(f"  - {pod_id}")
        return

    # Single pod launch
    pod_id = launcher.launch_training_pod(
        agent_range=args.agents,
        gpu_type=args.gpu_type,
    )

    if pod_id:
        print(f"\nLaunched pod: {pod_id}")
        print("Monitor with: runpodctl get pod")


if __name__ == "__main__":
    main()
