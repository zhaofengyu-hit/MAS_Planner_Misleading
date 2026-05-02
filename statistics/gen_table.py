"""Generate LaTeX tables from metrics_results.json."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

RESULTS_FILE = project_root / "tmp" / "table_data" / "metrics_results.json"


def load_results() -> dict:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, decimals: int = 2) -> str:
    """Format a numeric value or return '-' if missing."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _plan(data: dict, key: str) -> str:
    return _fmt(data.get("plan_metrics", {}).get(key))


def _exec_basic(data: dict, key: str) -> str:
    try:
        value = data["execution_metrics"]["basic_metrics"][key]["value"]
        return _fmt(value)
    except (KeyError, TypeError):
        return "-"


def _exec_token(data: dict) -> str:
    """Return avg_tokens_per_task in thousands (k), two decimal places."""
    try:
        value = data["execution_metrics"]["basic_metrics"]["avg_tokens_per_task"]["value"]
        return f"{float(value) / 1000:.2f}"
    except (KeyError, TypeError, ValueError):
        return "-"


def _plan_agent(data: dict, metric_key: str, agent_name: str) -> str:
    """Extract one agent's value from a per-agent plan metric dict."""
    try:
        val = data["plan_metrics"][metric_key]
        if isinstance(val, dict):
            v = val.get(agent_name, 0.0)
            return _fmt(v)
        return _fmt(val)
    except (KeyError, TypeError):
        return "-"


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

def table_subtask_and_verification(results: dict) -> str:
    """Table 1 — 过度分解 / 禁止分解 / 验证
    Columns: Type | AST | VIR (%) | DVTR (%) | Pass (%) | Token | Time (s)
    """
    rows = [
        ("Baseline",              "baseline"),
        ("Over-Fragmentation",    "over_fragmentation"),
        ("Under-Decomposition",   "under_decomposition"),
        ("Dependency Disruption", "dependency_disruption"),
    ]

    header = r"Type & AST & VIR (\%) & DVTR (\%) & Pass (\%) & Token (k) & Time (s) \\"
    col_spec = r"l >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X"

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for label, seed_id in rows:
        d = results.get(seed_id, {})
        ast   = _plan(d, "avg_subtasks_per_task")
        vir   = _plan(d, "verification_intent_ratio")
        dvtr  = _plan(d, "dedicated_verification_tasks_ratio")
        pass_ = _exec_basic(d, "success_rate")
        token = _exec_token(d)
        time_ = _exec_basic(d, "avg_time_per_task")
        lines.append(rf"{label} & {ast} & {vir} & {dvtr} & {pass_} & {token} & {time_} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def table_agent_assignment_exclusion(results: dict) -> str:
    """Table 2 — Over Assignment (Terminal) / Exclusion (Web)
    Columns: Type | APR | AMR | Pass (%) | Token | Time (s)
    Baseline uses multirow for Pass/Token/Time.
    """
    baseline_id   = "baseline"
    overassign_id = "over_assignment"
    exclusion_id  = "agent_exclusion"

    col_spec = r"l >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X"

    bl = results.get(baseline_id, {})
    oa = results.get(overassign_id, {})
    ex = results.get(exclusion_id, {})

    # Baseline shared execution metrics
    bl_pass  = _exec_basic(bl, "success_rate")
    bl_token = _exec_token(bl)
    bl_time  = _exec_basic(bl, "avg_time_per_task")

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"Type & APR (\%) & AMR (\%) & Pass (\%) & Token (k) & Time (s) \\")
    lines.append(r"\midrule")

    # Baseline: two agent rows, shared Pass/Token/Time via multirow
    bl_apr_t = _plan_agent(bl, "agent_participation_rate", "TerminalManager")
    bl_amr_t = _plan_agent(bl, "agent_task_mismatch_ratio", "TerminalManager")
    bl_apr_w = _plan_agent(bl, "agent_participation_rate", "WebSurfer")
    bl_amr_w = _plan_agent(bl, "agent_task_mismatch_ratio", "WebSurfer")

    lines.append(
        rf"Baseline (Terminal) & {bl_apr_t} & {bl_amr_t} & "
        rf"\multirow{{2}}{{*}}{{{bl_pass}}} & \multirow{{2}}{{*}}{{{bl_token}}} & \multirow{{2}}{{*}}{{{bl_time}}} \\"
    )
    lines.append(r"\cmidrule(lr){1-3}")
    lines.append(rf"Baseline (Web) & {bl_apr_w} & {bl_amr_w} & & & \\")
    lines.append(r"\midrule")

    # Attack rows
    oa_pass  = _exec_basic(oa, "success_rate")
    oa_token = _exec_token(oa)
    oa_time  = _exec_basic(oa, "avg_time_per_task")
    oa_apr   = _plan_agent(oa, "agent_participation_rate", "TerminalManager")
    oa_amr   = _plan_agent(oa, "agent_task_mismatch_ratio", "TerminalManager")
    lines.append(rf"Over Assign (Terminal) & {oa_apr} & {oa_amr} & {oa_pass} & {oa_token} & {oa_time} \\")

    ex_pass  = _exec_basic(ex, "success_rate")
    ex_token = _exec_token(ex)
    ex_time  = _exec_basic(ex, "avg_time_per_task")
    ex_apr   = _plan_agent(ex, "agent_participation_rate", "WebSurfer")
    ex_amr   = _plan_agent(ex, "agent_task_mismatch_ratio", "WebSurfer")
    lines.append(rf"Exclusion (Web) & {ex_apr} & {ex_amr} & {ex_pass} & {ex_token} & {ex_time} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def table_effort_midresult_prior(results: dict) -> str:
    """Table 3 — Overworking / Suppressed Intermediate Results / Prior Injection
    Columns: Type | EER (%) | MIRR (%) | AIR (%) | VCR (%) | PCR (%) | Pass (%) | Token | Time (s)
    """
    rows = [
        ("Baseline",              "baseline"),
        ("Overworking",           "overworking"),
        ("Suppressed Midresult",  "intermediate_output_suppression"),
        ("Prior Injection",       "planner_prior_misguidance"),
    ]

    col_spec = r"l" + r" >{\centering\arraybackslash}X" * 6

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        r"Type & EER (\%) & MIRR (\%) & VCR (\%) & Pass (\%) & Token (k) & Time (s) \\"
    )
    lines.append(r"\midrule")

    for label, seed_id in rows:
        d = results.get(seed_id, {})
        eer  = _plan(d, "subtasks_require_excessive_effort_ratio")
        mirr = _plan(d, "dependent_subtasks_no_clear_middle_result_ratio")
        vcr  = _plan(d, "vague_concretization_ratio")
        pass_ = _exec_basic(d, "success_rate")
        token = _exec_token(d)
        time_ = _exec_basic(d, "avg_time_per_task")
        lines.append(
            rf"{label} & {eer} & {mirr} & {vcr} & {pass_} & {token} & {time_} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def table_defense_comparison(results: dict) -> str:
    """Table 4 — Defense comparison: paired attack vs attack+defense rows.
    Groups: Baseline, Over-Fragmentation, Exclusion, Overworking.
    Columns: Type | AST | APR (%) | AMR (%) | EER (%) | Pass (%) | Token | Time (s)
    """
    col_spec = r"l" + r" >{\centering\arraybackslash}X" * 7

    def _row(d, label):
        ast   = _plan(d, "avg_subtasks_per_task")
        apr_w = _plan_agent(d, "agent_participation_rate", "WebSurfer")
        apr_t = _plan_agent(d, "agent_participation_rate", "TerminalManager")
        amr_w = _plan_agent(d, "agent_task_mismatch_ratio", "WebSurfer")
        amr_t = _plan_agent(d, "agent_task_mismatch_ratio", "TerminalManager")
        eer   = _plan(d, "subtasks_require_excessive_effort_ratio")
        pass_ = _exec_basic(d, "success_rate")
        token = _exec_token(d)
        time_ = _exec_basic(d, "avg_time_per_task")
        return rf"{label} & {ast} & {apr_w}/{apr_t} & {amr_w}/{amr_t} & {eer} & {pass_} & {token} & {time_} \\"

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"Type & AST & APR W/T (\%) & AMR W/T (\%) & EER (\%) & Pass (\%) & Token (k) & Time (s) \\")
    lines.append(r"\midrule")

    # Baseline pair
    bl   = results.get("baseline", {})
    bl_d = results.get("baseline_with_defense", {})
    lines.append(_row(bl,   "Baseline"))
    lines.append(_row(bl_d, "Baseline (Defense)"))
    lines.append(r"\midrule")

    # Over-Fragmentation pair
    of   = results.get("over_fragmentation", {})
    of_d = results.get("over_fragmentation_with_defense", {})
    lines.append(_row(of,   "Over-Fragmentation"))
    lines.append(_row(of_d, "Over-Fragmentation (Defense)"))
    lines.append(r"\midrule")

    # Exclusion pair (APR/AMR target: WebSurfer)
    ex   = results.get("agent_exclusion", {})
    ex_d = results.get("agent_exclusion_with_defense", {})
    lines.append(_row(ex,   "Exclusion"))
    lines.append(_row(ex_d, "Exclusion (Defense)"))
    lines.append(r"\midrule")

    # Overworking pair
    ow   = results.get("overworking", {})
    ow_d = results.get("overworking_with_defense", {})
    lines.append(_row(ow,   "Overworking"))
    lines.append(_row(ow_d, "Overworking (Defense)"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = load_results()

    print("% ===== Table 1: More/Less Subtasks + Verification =====")
    print(table_subtask_and_verification(results))
    print()

    print("% ===== Table 2: Over Assignment + Exclusion =====")
    print(table_agent_assignment_exclusion(results))
    print()

    print("% ===== Table 3: Overworking + Suppressed Midresult + Prior Injection =====")
    print(table_effort_midresult_prior(results))
    print()

    print("% ===== Table 4: Defense Comparison =====")
    print(table_defense_comparison(results))
    print()
