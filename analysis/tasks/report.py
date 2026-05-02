"""
Task: report
Output a LaTeX table row per platform with all analysis metrics.
Platforms with no LLM analysis yet are skipped.
"""
import logging

from ..analyzer import Analyzer, task

logger = logging.getLogger(__name__)

PLATFORM_LABELS = {
    "gpt_store":    "GPT Store",
    "coze":         "Coze",
    "wenxin":       "Baidu Wenxin",
    "apify":        "Apify",
    "yuanqi":       "Tencent Yuanqi",
    "agent_ai":     "Agent.ai",
    "relevance_ai": "Relevance AI",
}

# Display order
PLATFORM_ORDER = ["gpt_store", "coze", "wenxin", "yuanqi",
                  "apify", "agent_ai", "relevance_ai"]


def _pct(n, d) -> str:
    """Format as X.XX\\% (n/d), or -- if denominator is 0/None."""
    if not d:
        return "--"
    return f"{100.0 * n / d:.2f}\\% ({n:,}/{d:,})"


@task("report")
def run_report(
    self: Analyzer,
    platform: str | None = None,
    limit: int | None = None,
) -> None:
    # ── per-platform base counts ─────────────────────────────────────
    base_rows = self._conn.execute("""
        SELECT
            a.platform,
            COUNT(*)                                                AS total,
            SUM(CASE WHEN TRIM(COALESCE(a.description,''))=''
                     THEN 1 ELSE 0 END)                            AS empty_desc
        FROM agents a
        GROUP BY a.platform
    """).fetchall()
    base = {r[0]: {"total": r[1], "empty": r[2]} for r in base_rows}

    # ── LLM analysis counts (only platforms with data) ───────────────
    # All five metrics share the same denominator: analyzed (COUNT(*)).
    # "not meaningful": is_meaningful=0  OR  is_meaningful=1 AND all four=0
    # "no_X": COALESCE(has_X, 0) != 1  — covers both NULL (not evaluated)
    #         and explicitly 0; so is_meaningful=0 rows are naturally included.
    llm_rows = self._conn.execute("""
        SELECT
            a.platform,
            COUNT(*)                                                      AS analyzed,
            COALESCE(SUM(CASE
                WHEN aa.is_meaningful = 0 THEN 1
                WHEN aa.is_meaningful = 1
                 AND aa.has_capability  = 0
                 AND aa.has_input_spec  = 0
                 AND aa.has_output_spec = 0
                 AND aa.has_constraints = 0 THEN 1
                ELSE 0 END), 0)                                           AS not_meaningful,
            SUM(CASE WHEN COALESCE(aa.has_capability,  0) != 1 THEN 1 ELSE 0 END) AS no_cap,
            SUM(CASE WHEN COALESCE(aa.has_input_spec,  0) != 1 THEN 1 ELSE 0 END) AS no_in,
            SUM(CASE WHEN COALESCE(aa.has_output_spec, 0) != 1 THEN 1 ELSE 0 END) AS no_out,
            SUM(CASE WHEN COALESCE(aa.has_constraints, 0) != 1 THEN 1 ELSE 0 END) AS no_con
        FROM agents a
        JOIN agent_analysis aa ON a.id = aa.agent_id
        WHERE aa.is_meaningful IS NOT NULL
        GROUP BY a.platform
    """).fetchall()
    llm = {r[0]: r[1:] for r in llm_rows}

    # ── build table ──────────────────────────────────────────────────
    platforms = sorted(base.keys(),
                       key=lambda p: PLATFORM_ORDER.index(p)
                                     if p in PLATFORM_ORDER else 99)

    header = (
        "Platform & Collected & No Desc & Not Meaningful"
        " & No Capability & No Input & No Output & No Constraints \\\\"
    )
    print("\\begin{tabular}{lrrrrrrrr}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for plat in platforms:
        if plat not in llm:
            continue  # skip unanalyzed platforms

        b = base[plat]
        analyzed, not_m, no_cap, no_in, no_out, no_con = llm[plat]

        label = PLATFORM_LABELS.get(plat, plat)
        total = b["total"]
        empty = b["empty"]

        cols = [
            label,
            f"{total:,}",
            _pct(empty,   total),
            _pct(not_m,   analyzed),
            _pct(no_cap,  analyzed),
            _pct(no_in,   analyzed),
            _pct(no_out,  analyzed),
            _pct(no_con,  analyzed),
        ]
        print(" & ".join(cols) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
