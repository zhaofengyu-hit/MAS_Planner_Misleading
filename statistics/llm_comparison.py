"""LLM comparison: compute plan metrics and generate LaTeX table.

Usage:
    python statistics/llm_comparison.py compute   # compute plan metrics only
    python statistics/llm_comparison.py table     # print LaTeX table only
    python statistics/llm_comparison.py           # compute then print table
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

# { llm_display_name: { attack_label: variant_name } }
LLM_VARIANT_MAP: dict[str, dict[str, str]] = {
    "GPT-5": {
        "Baseline":           "baseline_gpt_5",
        "Over-Fragmentation": "over_fragmentation_gpt_5",
        "Exclusion":          "agent_exclusion_gpt_5",
        "Overworking":        "overworking_gpt_5",
    },
    "GPT-5-mini": {
        "Baseline":           "baseline_gpt_5_mini",
        "Over-Fragmentation": "over_fragmentation_gpt_5_mini",
        "Exclusion":          "agent_exclusion_gpt_5_mini",
        "Overworking":        "overworking_gpt_5_mini",
    },
    "Kimi": {
        "Baseline":           "baseline_kimi",
        "Over-Fragmentation": "over_fragmentation_kimi",
        "Exclusion":          "agent_exclusion_kimi",
        "Overworking":        "overworking_kimi",
    },
    "Qwen": {
        "Baseline":           "baseline_qwen",
        "Over-Fragmentation": "over_fragmentation_qwen",
        "Exclusion":          "agent_exclusion_qwen",
        "Overworking":        "overworking_qwen",
    },
}

RESULTS_FILE = project_root / "tmp" / "table_data" / "metrics_results_llm.json"


# ---------------------------------------------------------------------------
# Compute helpers
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


def _compute_plan_metrics(variant: str, existing: dict, checkpoint: dict) -> dict:
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
    all_variants = {
        variant
        for attacks in LLM_VARIANT_MAP.values()
        for variant in attacks.values()
    }

    checkpoint = _load_checkpoint()
    print(f"Loaded checkpoint with {len(checkpoint)} entries.\n")

    for variant in sorted(all_variants):
        print(f"=== {variant} ===")

        if variant not in checkpoint:
            checkpoint[variant] = {"plan_metrics": {}, "plan_metrics_done": False}
            _save(checkpoint)

        entry = checkpoint[variant]

        if entry.get("plan_metrics_done"):
            print("  [plan] already done, skipping")
        else:
            entry["plan_metrics"] = _compute_plan_metrics(
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


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def table_llm_comparison(results: dict) -> str:
    """Table — per-LLM plan metrics under baseline and three attacks.

    Columns: LLM (multirow 4) | Type | AST | APR W/T (%) | AMR W/T (%) | EER (%)
    """
    llm_names = list(LLM_VARIANT_MAP.keys())
    col_spec  = r"l l" + r" >{\centering\arraybackslash}X" * 4

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"LLM & Type & AST & APR W/T (\%) & AMR W/T (\%) & EER (\%) \\")
    lines.append(r"\midrule")

    for i, llm in enumerate(llm_names):
        variant_map  = LLM_VARIANT_MAP[llm]
        attack_order = list(variant_map.keys())

        for j, attack in enumerate(attack_order):
            variant = variant_map.get(attack, "")
            d       = results.get(variant, {}) if variant else {}

            ast   = _plan(d, "avg_subtasks_per_task")
            apr_w = _plan_agent(d, "agent_participation_rate", "WebSurfer")
            apr_t = _plan_agent(d, "agent_participation_rate", "TerminalManager")
            amr_w = _plan_agent(d, "agent_task_mismatch_ratio", "WebSurfer")
            amr_t = _plan_agent(d, "agent_task_mismatch_ratio", "TerminalManager")
            eer   = _plan(d, "subtasks_require_excessive_effort_ratio")

            llm_cell = rf"\multirow{{4}}{{*}}{{{llm}}}" if j == 0 else ""
            lines.append(
                rf"{llm_cell} & {attack} & {ast} & {apr_w}/{apr_t} & {amr_w}/{amr_t} & {eer} \\"
            )

        if i < len(llm_names) - 1:
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
        print("% ===== Table: LLM Comparison =====")
        print(table_llm_comparison(results))
