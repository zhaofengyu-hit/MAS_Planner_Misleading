"""Seed table-generation checkpoints from pre-computed data in tmp/plans/.

Run this once before gen_table.py / llm_comparison.py / real_agent_desc.py
to avoid re-running LLM-based plan metric computation.

Writes:
    tmp/table_data/metrics_results.json          (for gen_table.py)
    tmp/table_data/metrics_results_llm.json      (for llm_comparison.py)
    tmp/table_data/metrics_results_real_agent_desc.json  (for real_agent_desc.py)
"""

import json
from pathlib import Path

project_root = Path(__file__).parent.parent
plans_dir = project_root / "tmp" / "plans"
table_dir = project_root / "tmp" / "table_data"
table_dir.mkdir(parents=True, exist_ok=True)


def load_plan(name: str) -> dict:
    p = plans_dir / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def seed_main():
    """Seed metrics_results.json for gen_table.py (13 main variants)."""
    VARIANTS = [
        "baseline", "over_fragmentation", "under_decomposition",
        "dependency_disruption", "over_assignment", "agent_exclusion",
        "intermediate_output_suppression", "overworking", "planner_prior_misguidance",
        "baseline_with_defense", "over_fragmentation_with_defense",
        "agent_exclusion_with_defense", "overworking_with_defense",
    ]
    out_file = table_dir / "metrics_results.json"
    checkpoint = json.loads(out_file.read_text()) if out_file.exists() else {}

    for v in VARIANTS:
        d = load_plan(v)
        if d is None:
            print(f"[SKIP] {v}: plan file missing")
            continue
        entry = checkpoint.setdefault(v, {
            "plan_metrics": {}, "execution_metrics": None,
            "plan_metrics_done": False, "execution_metrics_done": False,
        })
        if not entry.get("plan_metrics_done") and d.get("plan_metrics"):
            entry["plan_metrics"] = d["plan_metrics"]
            entry["plan_metrics_done"] = True
        if not entry.get("execution_metrics_done") and d.get("execution_metrics") is not None:
            entry["execution_metrics"] = d["execution_metrics"]
            entry["execution_metrics_done"] = True
        checkpoint[v] = entry

    out_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    print(f"Seeded {out_file}")


def seed_llm():
    """Seed metrics_results_llm.json for llm_comparison.py."""
    LLM_VARIANT_MAP = {
        "GPT-5":      {"Baseline": "baseline_gpt_5", "Over-Fragmentation": "over_fragmentation_gpt_5", "Exclusion": "agent_exclusion_gpt_5", "Overworking": "overworking_gpt_5"},
        "GPT-5-mini": {"Baseline": "baseline_gpt_5_mini", "Over-Fragmentation": "over_fragmentation_gpt_5_mini", "Exclusion": "agent_exclusion_gpt_5_mini", "Overworking": "overworking_gpt_5_mini"},
        "Kimi":       {"Baseline": "baseline_kimi", "Over-Fragmentation": "over_fragmentation_kimi", "Exclusion": "agent_exclusion_kimi", "Overworking": "overworking_kimi"},
        "Qwen":       {"Baseline": "baseline_qwen", "Over-Fragmentation": "over_fragmentation_qwen", "Exclusion": "agent_exclusion_qwen", "Overworking": "overworking_qwen"},
    }
    out_file = table_dir / "metrics_results_llm.json"
    checkpoint = json.loads(out_file.read_text()) if out_file.exists() else {}

    for llm, attacks in LLM_VARIANT_MAP.items():
        for attack, variant in attacks.items():
            d = load_plan(variant)
            if d is None:
                print(f"[SKIP] {variant}: plan file missing")
                continue
            entry = checkpoint.setdefault(variant, {"plan_metrics": {}, "plan_metrics_done": False})
            if not entry.get("plan_metrics_done") and d.get("plan_metrics"):
                entry["plan_metrics"] = d["plan_metrics"]
                entry["plan_metrics_done"] = True
            checkpoint[variant] = entry

    out_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    print(f"Seeded {out_file}")


def seed_real_agent_desc():
    """Seed metrics_results_real_agent_desc.json for real_agent_desc.py."""
    CONDITION_MAP = {
        "Baseline":           ("baseline_replace_desc",           "baseline_replace_desc_with_defense"),
        "Over-Fragmentation": ("over_fragmentation_replace_desc", "over_fragmentation_replace_desc_with_defense"),
        "Exclusion":          ("agent_exclusion_replace_desc",    "agent_exclusion_replace_desc_with_defense"),
        "Overworking":        ("overworking_replace_desc",        "overworking_replace_desc_with_defense"),
    }
    out_file = table_dir / "metrics_results_real_agent_desc.json"
    checkpoint = json.loads(out_file.read_text()) if out_file.exists() else {}

    for _, (v_no_def, v_def) in CONDITION_MAP.items():
        for variant in (v_no_def, v_def):
            d = load_plan(variant)
            if d is None:
                print(f"[SKIP] {variant}: plan file missing")
                continue
            entry = checkpoint.setdefault(variant, {"plan_metrics": {}, "plan_metrics_done": False})
            if not entry.get("plan_metrics_done") and d.get("plan_metrics"):
                entry["plan_metrics"] = d["plan_metrics"]
                entry["plan_metrics_done"] = True
            checkpoint[variant] = entry

    out_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    print(f"Seeded {out_file}")


if __name__ == "__main__":
    seed_main()
    seed_llm()
    seed_real_agent_desc()
    print("\nDone. You can now run gen_table.py / llm_comparison.py table / real_agent_desc.py table directly.")
