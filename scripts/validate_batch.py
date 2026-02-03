#!/usr/bin/env python3
"""
Validate Batch of Trained Agents

Script to run after training completes on RunPod.
Downloads models and runs 5-filter validation pipeline.

Usage:
    # After downloading models from RunPod:
    python scripts/validate_batch.py --models-dir ./models_batch2

    # Or with custom catalog:
    python scripts/validate_batch.py --models-dir ./models_batch2 --catalog ./data/catalog
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.run_validation import ValidationPipeline


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def discover_models(models_dir: Path) -> list:
    """Discover trained models in directory."""
    agents = []

    for path in models_dir.iterdir():
        if path.is_dir():
            # Check for model files
            final_model = path / f"{path.name}_final.zip"
            best_model = path / "best" / "best_model.zip"

            if final_model.exists() or best_model.exists():
                agents.append(path.name)
                print(f"  Found: {path.name}")

    return sorted(agents)


def main():
    parser = argparse.ArgumentParser(description="Validate batch of trained agents")

    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="./data/catalog",
        help="Path to data catalog",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        default=True,
        help="Skip paper trading filter (default: True)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specific agents to validate (default: all)",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    catalog_path = Path(args.catalog)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        sys.exit(1)

    if not catalog_path.exists():
        print(f"WARNING: Catalog not found locally: {catalog_path}")
        print("  Make sure to download catalog from RunPod if needed")

    print_header("BATCH VALIDATION PIPELINE")
    print(f"  Models: {models_dir}")
    print(f"  Catalog: {catalog_path}")
    print(f"  Output: {output_dir}")

    # Discover models
    print_header("DISCOVERING MODELS")
    agents = args.agents or discover_models(models_dir)

    if not agents:
        print("ERROR: No trained models found")
        sys.exit(1)

    print(f"\n  Total agents found: {len(agents)}")

    # Initialize pipeline
    print_header("INITIALIZING VALIDATION PIPELINE")
    pipeline = ValidationPipeline(
        models_dir=str(models_dir),
        catalog_path=str(catalog_path),
        output_dir=str(output_dir),
    )

    # Run validation
    print_header("RUNNING 5-FILTER VALIDATION")
    start_time = datetime.now()

    results = pipeline.run_full_pipeline(
        agent_ids=agents,
        skip_paper=args.skip_paper,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print summary
    print_header("VALIDATION COMPLETE")
    print(f"  Duration: {duration:.1f} seconds")

    final_agents = results.get("summary", {}).get("final_agents", [])
    print(f"  Input agents: {len(agents)}")
    print(f"  Validated agents: {len(final_agents)}")

    if final_agents:
        print("\n  Validated agents:")
        for agent in final_agents:
            print(f"    - {agent}")
    else:
        print("\n  WARNING: No agents passed all filters")
        print("  Review validation results for details")

    # Save summary
    summary_file = output_dir / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": start_time.isoformat(),
            "models_dir": str(models_dir),
            "input_agents": agents,
            "validated_agents": final_agents,
            "duration_seconds": duration,
        }, f, indent=2)

    print(f"\n  Summary saved: {summary_file}")

    return 0 if final_agents else 1


if __name__ == "__main__":
    sys.exit(main())
