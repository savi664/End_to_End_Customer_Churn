"""Command-line entry point for scheduled drift monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.drift_monitor import detect_drift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_drift_check(
    reference_path: str | None = None,
    current_path: str | None = None,
    output_dir: str | None = None,
    moderate_threshold: float = 0.10,
    significant_threshold: float = 0.20,
) -> dict[str, Any]:
    """Run drift detection and return the report summary."""
    logger.info("Starting drift check at %s", datetime.now(timezone.utc).isoformat())

    results = detect_drift(
        reference_path=Path(reference_path) if reference_path else None,
        current_path=Path(current_path) if current_path else None,
        output_dir=Path(output_dir) if output_dir else None,
        moderate_threshold=moderate_threshold,
        significant_threshold=significant_threshold,
    )

    logger.info("Drift summary:\n%s", json.dumps(results, indent=2))

    if results.get("significant_drift", 0) > 0:
        logger.warning("Significant drift detected.")
    else:
        logger.info("No significant drift detected.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSI drift monitoring.")
    parser.add_argument("--reference", help="Reference CSV path.")
    parser.add_argument("--current", help="Current CSV path.")
    parser.add_argument("--output", help="Directory for drift reports.")
    parser.add_argument(
        "--moderate-threshold",
        type=float,
        default=0.10,
        help="PSI value that marks moderate drift.",
    )
    parser.add_argument(
        "--significant-threshold",
        type=float,
        default=0.20,
        help="PSI value that marks significant drift.",
    )
    args = parser.parse_args()

    results = run_drift_check(
        reference_path=args.reference,
        current_path=args.current,
        output_dir=args.output,
        moderate_threshold=args.moderate_threshold,
        significant_threshold=args.significant_threshold,
    )

    sys.exit(1 if results.get("significant_drift", 0) > 0 else 0)


if __name__ == "__main__":
    main()
