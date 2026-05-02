"""Real agent descriptions experiment: attack effectiveness with real-world agent descriptions.

Two conditions per attack type: without defense and with defense.
Table layout (8 rows, paired by condition):
    Baseline / Baseline+Defense
    Over-Fragmentation / Over-Fragmentation+Defense
    Exclusion / Exclusion+Defense
    Overworking / Overworking+Defense

Plan metrics only (no execution metrics): AST | APR W/T (%) | AMR W/T (%) | EER (%)

Usage:
    python statistics/real_agent_desc.py compute   # compute metrics only
    python statistics/real_agent_desc.py table     # print LaTeX table only
    python statistics/real_agent_desc.py           # compute then print table
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from statistics.plan_parser import PlanMetricsCalculator, PlanMetricType

PLANS_DIR = project_root / "tmp" / "plans"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# { condition_label: (no_defense_variant, defense_variant) }
CONDITION_MAP: dict[str, tuple[str, str]] = {
    "Baseline":           ("baseline_replace_desc",           "baseline_replace_desc_with_defense"),
    "Over-Fragmentation": ("over_fragmentation_replace_desc", "over_fragmentation_replace_desc_with_defense"),
    "Exclusion":          ("agent_exclusion_replace_desc",    "agent_exclusion_replace_desc_with_defense"),
    "Overworking":        ("overworking_replace_desc",        "overworking_replace_desc_with_defense"),
}

RESULTS_FILE = project_root / "tmp" / "table_data" / "metrics_results_real_agent_desc.json"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Compute helpers
# ---------------------------------------------------------------------------

def _compute_variant_metrics(variant: str, existing: dict, checkpoint: dict) -> dict:
    plan_file = PLANS_DIR / f"{variant}.json"
    if not plan_file.exists():
        print(f"  [plan] file not found: {plan_file}")
        return existing

    with open(plan_file, encoding="utf-8") as f:
        plan_results = json.load(f).get("plan_results")
    if not plan_results:
        print(f"  [plan] no plan_results in {plan_file.name}")
        return existing

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


def run_compute() -> None:
    checkpoint = _load_checkpoint()
    print(f"Loaded checkpoint with {len(checkpoint)} entries.\n")

    all_variants: list[tuple[str, str]] = []
    for label, (v_no_def, v_def) in CONDITION_MAP.items():
        all_variants.append((label, v_no_def))
        all_variants.append((f"{label}+Defense", v_def))

    for label, variant in all_variants:
        print(f"=== [{label}]  variant={variant} ===")

        if variant not in checkpoint:
            checkpoint[variant] = {"plan_metrics": {}, "plan_metrics_done": False}
            _save(checkpoint)

        entry = checkpoint[variant]

        if entry.get("plan_metrics_done"):
            print("  [plan] already done, skipping")
        else:
            entry["plan_metrics"] = _compute_variant_metrics(
                variant, entry.get("plan_metrics", {}), checkpoint
            )
            entry["plan_metrics_done"] = True
            checkpoint[variant] = entry
            _save(checkpoint)

        print()

    print(f"Results written to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _plan(data: dict, key: str) -> str:
    return _fmt(data.get("plan_metrics", {}).get(key))


def _plan_agent(data: dict, metric_key: str, agent_name: str) -> str:
    try:
        val = data["plan_metrics"][metric_key]
        if isinstance(val, dict):
            return _fmt(val.get(agent_name, 0.0))
        return _fmt(val)
    except (KeyError, TypeError):
        return "-"


def _row(results: dict, variant: str, label: str) -> str:
    d     = results.get(variant, {}) if variant else {}
    ast   = _plan(d, "avg_subtasks_per_task")
    apr_w = _plan_agent(d, "agent_participation_rate", "WebSurfer")
    apr_t = _plan_agent(d, "agent_participation_rate", "TerminalManager")
    amr_w = _plan_agent(d, "agent_task_mismatch_ratio", "WebSurfer")
    amr_t = _plan_agent(d, "agent_task_mismatch_ratio", "TerminalManager")
    eer   = _plan(d, "subtasks_require_excessive_effort_ratio")
    return rf"{label} & {ast} & {apr_w}/{apr_t} & {amr_w}/{amr_t} & {eer} \\"


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def table_real_agent_desc(results: dict) -> str:
    """Real agent descriptions table.

    Columns (5):
        Condition | AST | APR W/T (%) | AMR W/T (%) | EER (%)

    Eight rows in four paired groups separated by \\midrule.
    """
    col_spec = r"l" + r" >{\centering\arraybackslash}X" * 4

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"Condition & AST & APR W/T (\%) & AMR W/T (\%) & EER (\%) \\")
    lines.append(r"\midrule")

    conditions = list(CONDITION_MAP.items())
    for i, (label, (v_no_def, v_def)) in enumerate(conditions):
        lines.append(_row(results, v_no_def, label))
        lines.append(_row(results, v_def, f"{label}+Defense"))
        if i < len(conditions) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("compute", "both"):
        run_compute()

    if mode in ("table", "both"):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print("% ===== Table: Real Agent Descriptions =====")
        print(table_real_agent_desc(results))
