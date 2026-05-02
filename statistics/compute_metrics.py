"""Compute plan metrics and execution metrics for all experiment variants.

Usage:
    python statistics/compute_metrics.py

Reads:
    tmp/plans/<variant>.json        -- plan_results (from scripts/fetch_plans_new.py)
    tmp/logs/<variant>/             -- execution logs (from test/AttackTest/main.py)

Writes:
    tmp/table_data/metrics_results.json

Supports resume: already-computed metrics are skipped on restart.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from statistics.plan_parser import PlanMetricsCalculator, PlanMetricType
from statistics.log_parser import get_metrics_report

PLANS_DIR = project_root / "tmp" / "plans"
LOGS_DIR = project_root / "tmp" / "logs"
OUTPUT_FILE = project_root / "tmp" / "table_data" / "metrics_results.json"

VARIANTS = [
    "baseline",
    "over_fragmentation",
    "under_decomposition",
    "dependency_disruption",
    "over_assignment",
    "agent_exclusion",
    "intermediate_output_suppression",
    "overworking",
    "planner_prior_misguidance",
    "baseline_with_defense",
    "over_fragmentation_with_defense",
    "agent_exclusion_with_defense",
    "overworking_with_defense",
]

def _load_checkpoint() -> dict:
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compute_execution_metrics(variant: str) -> dict | None:
    log_dir = LOGS_DIR / variant
    if not log_dir.exists():
        print(f"  [exec] log dir not found: {log_dir}")
        return None
    try:
        report = get_metrics_report(str(log_dir))
        print(f"  [exec] done")
        return report
    except Exception as e:
        print(f"  [exec] ERROR: {e}")
        return None


def _compute_plan_metrics(
    plan_results: dict, existing: dict, checkpoint: dict, variant: str
) -> dict:
    calculator = PlanMetricsCalculator(plan_results)
    metrics = dict(existing)

    for metric_type in PlanMetricType:
        name = metric_type.value
        if name in metrics:
            print(f"  [plan] {name}: cached")
            continue
        try:
            result = calculator.calculate_metric(metric_type)
            metrics[name] = result[name]
            print(f"  [plan] {name}: {metrics[name]}")
        except Exception as e:
            metrics[name] = None
            print(f"  [plan] {name}: ERROR ({e})")

        checkpoint[variant]["plan_metrics"] = metrics
        _save(checkpoint)

    return metrics


def main():
    print(f"Processing {len(VARIANTS)} variant(s)\n")
    checkpoint = _load_checkpoint()

    for variant in VARIANTS:
        print(f"=== {variant} ===")

        if variant not in checkpoint:
            checkpoint[variant] = {
                "plan_metrics": {},
                "execution_metrics": None,
                "plan_metrics_done": False,
                "execution_metrics_done": False,
            }
            _save(checkpoint)

        entry = checkpoint[variant]

        if entry.get("execution_metrics_done"):
            print("  [exec] already done")
        else:
            result = _compute_execution_metrics(variant)
            entry["execution_metrics"] = result
            if result is not None:
                entry["execution_metrics_done"] = True
            checkpoint[variant] = entry
            _save(checkpoint)

        if entry.get("plan_metrics_done"):
            print("  [plan] already done")
        else:
            plan_file = PLANS_DIR / f"{variant}.json"
            if not plan_file.exists():
                print(f"  [plan] plan file not found: {plan_file}")
                print()
                continue
            with open(plan_file, encoding="utf-8") as f:
                plan_results = json.load(f).get("plan_results")
            if not plan_results:
                print(f"  [plan] no plan_results, skipping")
                continue
            existing = entry.get("plan_metrics", {})
            entry["plan_metrics"] = _compute_plan_metrics(
                plan_results, existing, checkpoint, variant
            )
            entry["plan_metrics_done"] = True
            checkpoint[variant] = entry
            _save(checkpoint)

        print()

    print(f"Done. Results at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
