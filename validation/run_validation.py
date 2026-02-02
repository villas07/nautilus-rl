#!/usr/bin/env python3
"""
Validation Pipeline Runner

Runs the complete 5-filter validation pipeline on trained agents.

Usage:
    python run_validation.py                     # Run all filters
    python run_validation.py --filter basic      # Run specific filter
    python run_validation.py --agents agent_001  # Validate specific agents
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import json

import structlog

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.filter_1_basic import BasicMetricsFilter, BasicMetricsCriteria
from validation.filter_2_cross_val import CrossValidationFilter
from validation.filter_3_diversity import DiversityFilter, DiversityCriteria
from validation.filter_4_walkforward import WalkForwardFilter
from validation.filter_5_paper import PaperTradingFilter

logger = structlog.get_logger()


class ValidationPipeline:
    """
    Complete validation pipeline for RL agents.

    Runs 5 filters in sequence:
    1. Basic Metrics (Sharpe > 1.5, DD < 15%, WR > 50%)
    2. Cross-Validation (performance on unseen instruments)
    3. Diversity (correlation < 0.5 between agents)
    4. Walk-Forward (out-of-sample 2024 data)
    5. Paper Trading (2-4 weeks live paper)

    Expected reduction: 500 agents -> ~30-50 validated agents
    """

    def __init__(
        self,
        models_dir: str = "/app/models",
        catalog_path: str = "/app/data/catalog",
        output_dir: str = "/app/validation",
    ):
        """
        Initialize pipeline.

        Args:
            models_dir: Directory containing trained models.
            catalog_path: Path to data catalog.
            output_dir: Directory for validation results.
        """
        self.models_dir = Path(models_dir)
        self.catalog_path = catalog_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize filters
        self.filter_1 = BasicMetricsFilter(
            models_dir=models_dir,
            catalog_path=catalog_path,
        )
        self.filter_2 = CrossValidationFilter(
            models_dir=models_dir,
            catalog_path=catalog_path,
        )
        self.filter_3 = DiversityFilter(
            models_dir=models_dir,
            catalog_path=catalog_path,
        )
        self.filter_4 = WalkForwardFilter(
            models_dir=models_dir,
            catalog_path=catalog_path,
        )
        self.filter_5 = PaperTradingFilter(
            models_dir=models_dir,
        )

        # Results storage
        self.results: Dict[str, Any] = {}

    def discover_agents(self) -> List[str]:
        """Discover all trained agents."""
        agents = []

        for path in self.models_dir.iterdir():
            if path.is_dir():
                # Check for model file
                model_file = path / f"{path.name}_final.zip"
                best_model = path / "best" / "best_model.zip"

                if model_file.exists() or best_model.exists():
                    agents.append(path.name)

        logger.info(f"Discovered {len(agents)} trained agents")
        return sorted(agents)

    def run_filter_1(
        self,
        agent_ids: List[str],
    ) -> tuple:
        """
        Run Filter 1: Basic Metrics.

        Returns (passed_agents, metrics_dict).
        """
        logger.info("=" * 60)
        logger.info("FILTER 1: Basic Metrics Validation")
        logger.info("=" * 60)

        results = self.filter_1.validate_batch(agent_ids)

        passed = [r.agent_id for r in results if r.passed]
        metrics = {r.agent_id: r.metrics for r in results}

        logger.info(f"Filter 1: {len(passed)}/{len(agent_ids)} agents passed")

        self.results["filter_1"] = {
            "input_count": len(agent_ids),
            "output_count": len(passed),
            "passed_agents": passed,
            "all_results": [
                {
                    "agent_id": r.agent_id,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "failures": r.failures,
                }
                for r in results
            ],
        }

        return passed, metrics

    def run_filter_2(
        self,
        agent_ids: List[str],
        training_metrics: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """
        Run Filter 2: Cross-Validation.

        Returns list of passed agents.
        """
        logger.info("=" * 60)
        logger.info("FILTER 2: Cross-Validation")
        logger.info("=" * 60)

        passed = []
        all_results = []

        for agent_id in agent_ids:
            metrics = training_metrics.get(agent_id)
            result = self.filter_2.validate(agent_id, metrics)
            all_results.append(result)

            if result.get("passed", False):
                passed.append(agent_id)

        logger.info(f"Filter 2: {len(passed)}/{len(agent_ids)} agents passed")

        self.results["filter_2"] = {
            "input_count": len(agent_ids),
            "output_count": len(passed),
            "passed_agents": passed,
            "all_results": all_results,
        }

        return passed

    def run_filter_3(
        self,
        agent_ids: List[str],
        metrics: Dict[str, Dict[str, float]],
        target_agents: int = 50,
    ) -> List[str]:
        """
        Run Filter 3: Diversity Selection.

        Returns list of diverse agents.
        """
        logger.info("=" * 60)
        logger.info("FILTER 3: Diversity Filter")
        logger.info("=" * 60)

        self.filter_3.criteria.target_agents = target_agents
        selected = self.filter_3.filter(agent_ids, metrics)

        # Get diversity stats
        diversity_stats = self.filter_3.compute_ensemble_diversity(selected)

        logger.info(f"Filter 3: Selected {len(selected)} diverse agents")
        logger.info(f"  Avg correlation: {diversity_stats.get('avg_correlation', 0):.3f}")
        logger.info(f"  Max correlation: {diversity_stats.get('max_correlation', 0):.3f}")

        self.results["filter_3"] = {
            "input_count": len(agent_ids),
            "output_count": len(selected),
            "passed_agents": selected,
            "diversity_stats": diversity_stats,
        }

        return selected

    def run_filter_4(
        self,
        agent_ids: List[str],
        training_metrics: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """
        Run Filter 4: Walk-Forward Validation.

        Returns list of passed agents.
        """
        logger.info("=" * 60)
        logger.info("FILTER 4: Walk-Forward Validation")
        logger.info("=" * 60)

        passed = []
        all_results = []

        for agent_id in agent_ids:
            metrics = training_metrics.get(agent_id)
            result = self.filter_4.validate(agent_id, metrics)
            all_results.append(result)

            if result.get("passed", False):
                passed.append(agent_id)

        logger.info(f"Filter 4: {len(passed)}/{len(agent_ids)} agents passed")

        self.results["filter_4"] = {
            "input_count": len(agent_ids),
            "output_count": len(passed),
            "passed_agents": passed,
            "all_results": all_results,
        }

        return passed

    def run_filter_5(
        self,
        agent_ids: List[str],
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Run Filter 5: Paper Trading.

        If session_id provided, validates from existing session.
        Otherwise, starts new paper trading session.

        Returns list of passed agents.
        """
        logger.info("=" * 60)
        logger.info("FILTER 5: Paper Trading Validation")
        logger.info("=" * 60)

        if session_id:
            # Validate from existing session
            results = self.filter_5.validate_from_session(session_id)
            passed = [
                agent_id for agent_id, result in results.items()
                if result.get("passed", False)
            ]

            self.results["filter_5"] = {
                "input_count": len(agent_ids),
                "output_count": len(passed),
                "passed_agents": passed,
                "session_id": session_id,
                "all_results": results,
            }
        else:
            # Start new paper trading session
            session = self.filter_5.start_paper_trading(agent_ids)
            passed = []  # Will be determined after paper trading completes

            self.results["filter_5"] = {
                "input_count": len(agent_ids),
                "status": "paper_trading_started",
                "session_id": session["session_id"],
                "message": "Run again with --paper-session after 2-4 weeks",
            }

            logger.info(
                f"Started paper trading session: {session['session_id']}"
            )
            logger.info("Run validation again with --paper-session after 2-4 weeks")

        return passed

    def run_full_pipeline(
        self,
        agent_ids: Optional[List[str]] = None,
        skip_paper: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete validation pipeline.

        Args:
            agent_ids: List of agents to validate (or discover all).
            skip_paper: Skip filter 5 (paper trading).

        Returns:
            Complete validation results.
        """
        start_time = datetime.now()

        # Discover or use provided agents
        if agent_ids is None:
            agent_ids = self.discover_agents()

        if not agent_ids:
            logger.warning("No agents to validate")
            return {"error": "No agents found"}

        logger.info(f"Starting validation of {len(agent_ids)} agents")

        # Filter 1: Basic Metrics
        passed_1, metrics = self.run_filter_1(agent_ids)

        if not passed_1:
            logger.warning("No agents passed Filter 1")
            return self._finalize_results(start_time)

        # Filter 2: Cross-Validation
        passed_2 = self.run_filter_2(passed_1, metrics)

        if not passed_2:
            logger.warning("No agents passed Filter 2")
            return self._finalize_results(start_time)

        # Filter 3: Diversity
        passed_3 = self.run_filter_3(passed_2, metrics)

        if not passed_3:
            logger.warning("No agents passed Filter 3")
            return self._finalize_results(start_time)

        # Filter 4: Walk-Forward
        passed_4 = self.run_filter_4(passed_3, metrics)

        if not passed_4:
            logger.warning("No agents passed Filter 4")
            return self._finalize_results(start_time)

        # Filter 5: Paper Trading
        if not skip_paper:
            passed_5 = self.run_filter_5(passed_4)
        else:
            passed_5 = passed_4
            self.results["filter_5"] = {
                "skipped": True,
                "passed_agents": passed_4,
            }

        return self._finalize_results(start_time)

    def _finalize_results(self, start_time: datetime) -> Dict[str, Any]:
        """Finalize and save results."""
        end_time = datetime.now()

        self.results["summary"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "final_agents": self._get_final_agents(),
        }

        # Save results
        results_file = self.output_dir / f"validation_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

        # Print summary
        self._print_summary()

        return self.results

    def _get_final_agents(self) -> List[str]:
        """Get final list of validated agents."""
        for filter_name in ["filter_5", "filter_4", "filter_3", "filter_2", "filter_1"]:
            if filter_name in self.results:
                filter_result = self.results[filter_name]
                if "passed_agents" in filter_result:
                    return filter_result["passed_agents"]
        return []

    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION PIPELINE SUMMARY")
        print("=" * 60)

        for i in range(1, 6):
            filter_name = f"filter_{i}"
            if filter_name in self.results:
                result = self.results[filter_name]
                input_count = result.get("input_count", "?")
                output_count = result.get("output_count", result.get("status", "?"))
                print(f"  Filter {i}: {input_count} -> {output_count}")

        final_agents = self._get_final_agents()
        print(f"\n  Final validated agents: {len(final_agents)}")

        if final_agents:
            print("\n  Agents:")
            for agent in final_agents[:10]:  # Show first 10
                print(f"    - {agent}")
            if len(final_agents) > 10:
                print(f"    ... and {len(final_agents) - 10} more")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run validation pipeline")

    # Resolve default paths relative to project root
    project_root = Path(__file__).parent.parent
    default_models = str(project_root / "models")
    default_catalog = str(project_root / "data" / "catalog")
    default_output = str(project_root / "validation" / "results")

    parser.add_argument(
        "--models-dir",
        default=default_models,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--catalog",
        default=default_catalog,
        help="Path to data catalog",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help="Output directory for results",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specific agents to validate",
    )
    parser.add_argument(
        "--filter",
        choices=["basic", "crossval", "diversity", "walkforward", "paper", "all"],
        default="all",
        help="Which filter to run",
    )
    parser.add_argument(
        "--paper-session",
        help="Paper trading session ID to validate from",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="Skip paper trading filter",
    )
    parser.add_argument(
        "--target-agents",
        type=int,
        default=50,
        help="Target number of diverse agents (default: 50)",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ValidationPipeline(
        models_dir=args.models_dir,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
    )

    # Run validation
    if args.filter == "all":
        results = pipeline.run_full_pipeline(
            agent_ids=args.agents,
            skip_paper=args.skip_paper,
        )
    else:
        # Run individual filter
        agent_ids = args.agents or pipeline.discover_agents()

        if args.filter == "basic":
            passed, _ = pipeline.run_filter_1(agent_ids)
        elif args.filter == "crossval":
            passed_1, metrics = pipeline.run_filter_1(agent_ids)
            passed = pipeline.run_filter_2(passed_1, metrics)
        elif args.filter == "diversity":
            passed_1, metrics = pipeline.run_filter_1(agent_ids)
            passed = pipeline.run_filter_3(passed_1, metrics, args.target_agents)
        elif args.filter == "walkforward":
            passed_1, metrics = pipeline.run_filter_1(agent_ids)
            passed = pipeline.run_filter_4(passed_1, metrics)
        elif args.filter == "paper":
            passed = pipeline.run_filter_5(agent_ids, args.paper_session)


if __name__ == "__main__":
    main()
