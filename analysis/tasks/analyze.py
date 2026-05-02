"""
Task: analyze
Single LLM call that judges all five dimensions for one agent.

  is_meaningful   — is the description readable, human-intentional text?
  has_capability  — does it mention what the agent can DO?
  has_input_spec  — does it describe what INPUT the user should provide?
  has_output_spec — does it describe what OUTPUT the agent produces?
  has_constraints — does it mention any explicit LIMITATIONS or restrictions?

Short-circuit: if is_meaningful=false, the four spec fields are stored as NULL
(not applicable), not as 0. This keeps them distinguishable from "evaluated,
not present".

Prerequisite: run fill_desc first.

Targets agents where has_description=1 AND is_meaningful IS NULL.
Use --platform / --limit to control the subset.
"""
import logging

from pydantic import BaseModel, Field

from ..analyzer import Analyzer, task

logger = logging.getLogger(__name__)

_SYSTEM = """
You are analyzing agent/bot descriptions collected from AI marketplaces.
Descriptions may be in any language (English, Chinese, etc.).

Judge five things about the description. Answer each with true or false.

is_meaningful
  Is this readable, human-intentional text — even if extremely vague?
  Answer FALSE only for pure noise: random characters, keyboard mashing,
  meaningless repetition (e.g. "bababababab", "asdfghjkl", "哈哈哈哈哈哈哈哈").
  Vague text such as "这是一个AI助手" or "Help you." is TRUE.

If is_meaningful is false, set the remaining four fields to false as well
(they will be discarded in storage).

has_capability
  Does the description mention what the agent can DO — its task, function,
  or purpose? Even a vague mention counts ("写作助手", "helps you code").
  Do NOT count pure performance claims ("reduces latency by 50%") as capability.

has_input_spec
  Does the description mention what INPUT the user should provide?
  This includes: content type (text, file, URL, SQL query…),
  delivery format (paste here, upload a file…), or any input requirement.
  Only count the agent's own input interface, not instructions to other systems.

has_output_spec
  Does the description mention what OUTPUT or result the agent produces?
  This includes: content type (report, summary, code, translation…) or
  delivery format (returned as text, saved to file…).
  Only count this agent's own deliverables.

has_constraints
  Does the description explicitly state any limitation, restriction, or
  scope boundary that applies to this agent?
  The constraint must be stated directly — do not infer it from capability
  descriptions. Examples: language restrictions, topic limits, rate limits,
  explicit "only supports X", explicit "cannot do Y".
""".strip()


class _Result(BaseModel):
    is_meaningful: bool = Field(
        description=(
            "True if the text is readable human-intentional content, even if vague. "
            "False ONLY for pure gibberish, random characters, or meaningless repetition."
        )
    )
    has_capability: bool = Field(
        description="True if the description mentions what the agent can do, even vaguely."
    )
    has_input_spec: bool = Field(
        description="True if the description mentions what input the user should provide."
    )
    has_output_spec: bool = Field(
        description="True if the description mentions what output the agent produces."
    )
    has_constraints: bool = Field(
        description=(
            "True if the description explicitly states a limitation, restriction, "
            "or scope boundary. Do not infer from capabilities."
        )
    )


def _user_prompt(agent: dict) -> str:
    name = (agent["name"] or "").strip()
    desc = (agent["description"] or "").strip()
    return f"Name: {name!r}\nDescription: {desc!r}"


@task("analyze")
def run_analyze(
    self: Analyzer,
    platform: str | None = None,
    limit: int | None = None,
) -> None:
    self._ensure_analysis_table()

    where = "WHERE aa.has_description = 1 AND aa.is_meaningful IS NULL"
    params: list = []
    if platform:
        where += " AND a.platform = ?"
        params.append(platform)
    limit_clause = f"LIMIT {limit}" if limit else ""

    agents = [
        {"id": r[0], "platform": r[1], "name": r[2], "description": r[3]}
        for r in self._conn.execute(
            f"""SELECT a.id, a.platform, a.name, a.description
                FROM agents a
                JOIN agent_analysis aa ON a.id = aa.agent_id
                {where}
                ORDER BY RANDOM()
                {limit_clause}""",
            params,
        ).fetchall()
    ]

    if not agents:
        logger.info("Nothing to process.")
        return

    logger.info("Analyzing %d agents …", len(agents))
    done = 0
    for agent in agents:
        result = self._call_structured(_SYSTEM, _user_prompt(agent), _Result)
        if result.is_meaningful:
            self._upsert_analysis(
                agent["id"],
                is_meaningful=1,
                has_capability=int(result.has_capability),
                has_input_spec=int(result.has_input_spec),
                has_output_spec=int(result.has_output_spec),
                has_constraints=int(result.has_constraints),
            )
        else:
            # has_* remain NULL: "not applicable", not "evaluated as absent"
            self._upsert_analysis(agent["id"], is_meaningful=0)
        done += 1
        if done % 100 == 0:
            logger.info("Progress: %d / %d", done, len(agents))

    logger.info("Done. Analyzed %d agents.", done)
