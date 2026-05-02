"""Fetch plans for a single experiment variant and save results to disk.

Usage:
    python scripts/fetch_plans_new.py --name <variant> --desc <description>
                                      [--planner <name>] [--use_defense]
                                      [--replace_worker_descs] [--overwrite]

variant name examples: baseline, attack_1, baseline_defense, attack_1_defense

Results are written to:
    tmp/plans/<name>.json

If the result file already exists, the run is skipped unless --overwrite is set.
Checkpoints (for resuming interrupted runs) are saved under:
    tmp/plan_cache/<name>/checkpoint.json

Available planners: gpt, deepseek, deepseek_ali, kimi, qwen
Defense options:
  --use_defense        Enable DescriptionNormalizer.
Worker description replacement:
  --replace_worker_descs  Load non-DB agent descriptions from
                          scripts/agent_descriptions_replace.json.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_WORKER_DESCS_FILE = Path(__file__).parent / "agent_descriptions_replace.json"
_PLANS_DIR = project_root / "tmp" / "plans"

from test.AttackTest.PlannerTest import get_plans


def _load_worker_descs() -> dict:
    with open(_WORKER_DESCS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _strip_null_bytes(obj):
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, dict):
        return {k: _strip_null_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_null_bytes(v) for v in obj]
    return obj


def fetch_plans_for_variant(
    name: str,
    desc: str,
    planner: str | None = None,
    use_defense: bool = False,
    worker_descs: dict | None = None,
    overwrite: bool = False,
) -> None:
    import asyncio

    out_path = _PLANS_DIR / f"{name}.json"
    if out_path.exists() and not overwrite:
        print(f"Already exists, skipping: {out_path}")
        return

    _PLANS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = str(project_root / "Benchmark" / "GAIA")
    cache_dir = str(project_root / "tmp" / "plan_cache" / name)

    kwargs = dict(
        use_defense=use_defense,
        cache_dir=cache_dir,
    )
    if planner is not None:
        kwargs["planner"] = planner
    if worker_descs is not None:
        kwargs["worker_descs"] = worker_descs

    plan_results = asyncio.run(get_plans(desc, data_dir, **kwargs))

    result = {
        "name": name,
        "plan_results": _strip_null_bytes(plan_results),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch plans for one experiment variant.")
    parser.add_argument("--name", type=str, required=True, help="Variant name (e.g. baseline, attack_1)")
    parser.add_argument("--desc", type=str, required=True, help="Agent description string")
    parser.add_argument("--planner", default=None, help="Planner model key (e.g. gpt, deepseek_ali, kimi, qwen)")
    parser.add_argument("--use_defense", action="store_true", default=False)
    parser.add_argument(
        "--replace_worker_descs",
        action="store_true",
        default=False,
        help="Replace non-DB worker descriptions from scripts/agent_descriptions_replace.json",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Re-run even if result file exists")
    args = parser.parse_args()

    worker_descs = _load_worker_descs() if args.replace_worker_descs else None

    print(f"Variant: {args.name}")
    print(f"  use_defense={args.use_defense}, planner={args.planner}")

    fetch_plans_for_variant(
        name=args.name,
        desc=args.desc,
        planner=args.planner,
        use_defense=args.use_defense,
        worker_descs=worker_descs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
