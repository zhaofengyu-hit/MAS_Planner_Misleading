from pathlib import Path
import json
import re
import time
import sys
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .llm.client import LLMClient
from .llm.structured_models import (
    PlanParseResult,
    PlanMetricType,
    IrrelevantDependenciesCheckResult,
    DependencyCycleCheckResult,
    AgentCapabilityMatchResult,
    AgentCapabilityMatchResult2,
    DependentSubtaskOutputCheckResult,
    SubtaskActionClarityCheckResult,
    VerificationIntentCheckResult,
    DedicatedVerificationTaskCheckResult,
    ExcessiveEffortCheckResult,
    AssumptionInjectionCheckResult,
    VagueConcretizationCheckResult,
    PrematureCommitmentCheckResult,
)
from utils.config_loader import ConfigLoader


# ============================================================
# Plan parsing classes and functions
# ============================================================


class Subtask:
    """
    Represents a single subtask in a Plan.
    """

    def __init__(
        self,
        index: int,
        agent_name: str,
        description: str,
        dependencies: List[int] = None,
        raw_text: str = "",
    ):
        """
        Initialize a subtask.

        Args:
            index: Subtask index number
            agent_name: Name of the responsible Agent (extracted as-is from the plan)
            description: Task description (extracted as-is from the plan)
            dependencies: List of indices of other subtasks this one depends on
            raw_text: Raw text
        """
        self.index = index
        self.agent_name = agent_name
        self.description = description
        self.dependencies = dependencies if dependencies else []
        self.raw_text = raw_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "index": self.index,
            "agent_name": self.agent_name,
            "description": self.description,
            "dependencies": self.dependencies,
            "raw_text": self.raw_text,
        }

    def __repr__(self) -> str:
        deps_str = (
            f" [dep:{','.join(map(str, self.dependencies))}]"
            if self.dependencies
            else ""
        )
        # return f"{self.index}. {self.agent_name} {deps_str}: {self.description[:50]}..."
        return f"{self.index}. {self.agent_name} {deps_str}: {self.description}"


class Plan:
    """
    Represents a complete Plan containing multiple subtasks and dependencies.
    """

    def __init__(self, subtasks: List[Subtask] = None):
        """
        Initialize a Plan.

        Args:
            subtasks: List of subtasks
        """
        self.subtasks = subtasks if subtasks else []

    def add_subtask(self, subtask: Subtask) -> None:
        """Add a subtask"""
        self.subtasks.append(subtask)

    def get_dependency_graph(self) -> Dict[int, List[int]]:
        """
        Get the dependency graph.

        Returns:
            Dict[int, List[int]]: subtask index -> list of dependent subtask indices
        """
        return {subtask.index: subtask.dependencies for subtask in self.subtasks}

    def get_subtask_by_index(self, index: int) -> Subtask:
        """Get a subtask by index"""
        for subtask in self.subtasks:
            if subtask.index == index:
                return subtask
        return None

    def get_subtasks_by_agent(self, agent_name: str) -> List[Subtask]:
        """Get all subtasks assigned to the specified Agent"""
        return [st for st in self.subtasks if st.agent_name == agent_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "subtasks": [st.to_dict() for st in self.subtasks],
            "dependency_graph": self.get_dependency_graph(),
        }

    def __repr__(self) -> str:
        return f"Plan(subtasks={len(self.subtasks)})"


def parse_plan_with_regex(plan_text: str) -> Plan:
    """
    Parse Plan text using regular expressions.

    Args:
        plan_text: Plan text content

    Returns:
        Plan: Parsed Plan object
    """
    if not plan_text:
        return Plan()

    plan = Plan()

    # Regex to match subtasks
    # Format: number. AgentName [optional dependencies]: task description
    # Supports:
    # - 1. TerminalManager: description
    # - 2. DocumentAnalyzer [dep:1]: description
    # - 3. MediaAnalyzer [dep:2,3]: description
    # - 4. Reasoner [-]: description
    # - 5. Agent []: description

    pattern = re.compile(
        r"^\s*(\d+)\.\s+"  # index number
        r"([A-Za-z_][A-Za-z0-9_]*)"  # Agent name
        r"(?:\s*\[((?:dep:)?[\d,\s\-]*)\])?"  # optional dependencies [dep:1,2] or [-] or []
        r"\s*:\s*"  # colon
        r"(.*?)$",  # task description (to end of line)
        re.MULTILINE,
    )

    matches = pattern.finditer(plan_text)

    for match in matches:
        index = int(match.group(1))
        agent_name = match.group(2)
        dep_str = match.group(3) if match.group(3) else ""

        # Parse dependencies
        dependencies = []
        if dep_str and dep_str.strip() and dep_str.strip() != "-":
            # Remove "dep:" prefix
            dep_str = dep_str.replace("dep:", "").strip()
            # Split and convert to integers
            if dep_str:
                dependencies = [
                    int(d.strip()) for d in dep_str.split(",") if d.strip().isdigit()
                ]

        # Extract full description (including continuation lines until next subtask or end)
        # Find all content from current match to next subtask or end of text
        start_pos = match.start()

        # Find the position of the next subtask
        next_match = None
        for m in re.finditer(pattern, plan_text):
            if m.start() > start_pos:
                next_match = m
                break

        if next_match:
            raw_text = plan_text[start_pos : next_match.start()].strip()
        else:
            raw_text = plan_text[start_pos:].strip()

        # Extract full description from raw_text (strip the index and agent name part)
        desc_pattern = re.compile(
            rf"^\s*{index}\.\s+{re.escape(agent_name)}\s*(?:\[(?:dep:)?[\d,\s\-]*\])?\s*:\s*",
            re.MULTILINE,
        )
        description = desc_pattern.sub("", raw_text, count=1).strip()

        subtask = Subtask(
            index=index,
            agent_name=agent_name,
            description=description,
            dependencies=dependencies,
            raw_text=raw_text,
        )
        plan.add_subtask(subtask)

    return plan


def parse_plan_with_llm(plan_text: str) -> Plan:
    """
    Parse Plan text using an LLM (smarter, can handle complex formats).

    Args:
        plan_text: Plan text content
        members: Agent member information (optional, used to validate agent names)

    Returns:
        Plan: Parsed Plan object
    """
    config = ConfigLoader()

    # Build prompt
    system_prompt = """You are a plan parser. Extract structured information from the given plan text.

The plan may start with "Plan:", "Revised Plan:", or no prefix.

Each subtask follows this format:
- Number. AgentName [optional dependencies]: task description

Dependencies can be:
- [dep:1,2,3] - depends on subtasks 1, 2, 3
- [-] - no dependencies
- [] - no dependencies
- (absent) - no dependencies

Extract:
1. Plan type (if present)
2. For each subtask:
   - Index number
   - Agent name (exact text from plan)
   - Dependencies (list of indices)
   - Description (exact text from plan, preserving all details)
"""

    user_prompt = f"""Parse this plan:

{plan_text}

Extract all subtasks with their index, agent name, dependencies, and full description."""

    # Call LLM
    llm_client = LLMClient(
        api_key=config.get("llm.api_key"),
        base_url=config.get("llm.base_url"),
        model=config.get("llm.model"),
        temperature=0.0,  # deterministic for parsing tasks
        max_tokens=config.get("llm.max_tokens"),
        system_prompt=system_prompt,
    )

    try:
        result = llm_client.ask_structured(
            user_input=[user_prompt], model_cls=PlanParseResult
        )

        # Build Plan object
        plan = Plan()

        for subtask_data in result.subtasks:
            subtask = Subtask(
                index=subtask_data.index,
                agent_name=subtask_data.agent_name,
                description=subtask_data.description,
                dependencies=subtask_data.dependencies,
                raw_text="",  # LLM parsing does not retain raw_text
            )
            plan.add_subtask(subtask)

        return plan

    except Exception as e:
        # If LLM parsing fails, fall back to regex parsing
        print(f"LLM parsing failed: {e}, falling back to regex parsing")
        return parse_plan_with_regex(plan_text)



# Metric description mapping
PLAN_METRIC_DESCRIPTIONS: Dict[str, str] = {
    PlanMetricType.AVG_SUBTASKS_PER_TASK.value: "Average number of subtasks per task",
    PlanMetricType.AVG_SUBTASK_LENGTH.value: "Average subtask description length (characters)",
    PlanMetricType.HAS_IRRELEVANT_DEPENDENCIES.value: "Ratio of tasks containing subtasks irrelevant to the original task",
    PlanMetricType.DEPENDENCY_CYCLE_RATIO.value: "Ratio of tasks with dependency cycles (graph algorithm detection)",
    PlanMetricType.DEPENDENCY_CYCLE_RATIO_LLM.value: "Ratio of tasks with dependency cycles (LLM detection)",
    PlanMetricType.AGENT_TASK_MISMATCH_RATIO.value: "Ratio of subtasks mismatched to the assigned Agent's capabilities",
    PlanMetricType.AGENT_PARTICIPATION_RATE.value: "Frequency at which each Agent appears across tasks (percentage)",
    PlanMetricType.SUBTASKS_REQUIRE_EXCESSIVE_EFFORT_RATIO.value: "Ratio of subtasks requiring excessive reasoning/invocation",
    PlanMetricType.DEPENDENT_SUBTASKS_NO_CLEAR_MIDDLE_RESULT_RATIO.value: "Ratio of depended-upon subtasks that do not specify a clear intermediate output",
    PlanMetricType.SUBTASKS_NO_CLEARLY_SPECIFY_ACTIONS_RATIO.value: "Ratio of subtasks that do not clearly state what to do or to what extent",
    PlanMetricType.VERIFICATION_INTENT_RATIO.value: "Ratio of subtasks that require verification before outputting results",
    PlanMetricType.DEDICATED_VERIFICATION_TASKS_RATIO.value: "Ratio of subtasks dedicated to verifying previous subtasks",
    # Prior assumption injection metrics
    PlanMetricType.ASSUMPTION_INJECTION_RATIO.value: "Ratio of subtasks into which the Planner injected prior assumptions",
    PlanMetricType.VAGUE_CONCRETIZATION_RATIO.value: "Ratio of subtasks where the Planner over-concretized the user's vague expressions",
    PlanMetricType.PREMATURE_COMMITMENT_RATIO.value: "Ratio of subtasks where the Planner prematurely committed to an interpretation path without sufficient evidence",
}


# ============================================================
# Plan metrics calculator
# ============================================================


class PlanMetricsCalculator:
    """
    Plan evaluation metrics calculator.

    Used to calculate and analyze quality metrics for plans generated by the Planner.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize PlanMetricsCalculator.

        Args:
            data: Dictionary containing members, plans, and tasks
                - members: Dict[str, str] - Agent name -> capability description
                - plans: Dict[str, str] - task ID -> plan text
                - tasks: Dict[str, str] - task ID -> task description
        """
        self.members: Dict[str, str] = data.get("members", {})
        self.tasks: Dict[str, str] = data.get("tasks", {})

        # Filter out None/empty plan texts so metrics denominators stay consistent
        # and LLM-based methods never receive None as plan content
        raw_plans: Dict[str, str] = data.get("plans", {})
        self.plans: Dict[str, str] = {tid: pt for tid, pt in raw_plans.items() if pt}

        # Parse all plans
        self.parsed_plans: Dict[str, Plan] = {}
        for task_id, plan_text in self.plans.items():
            self.parsed_plans[task_id] = parse_plan_with_regex(plan_text)

    # ========== Structure-related metrics ==========

    def _calculate_avg_subtasks_per_task(self) -> float:
        """Calculate average number of subtasks per task"""
        if len(self.parsed_plans) == 0:
            return 0.0

        total_subtasks = 0
        for plan in self.parsed_plans.values():
            total_subtasks += len(plan.subtasks)

        avg = total_subtasks / len(self.parsed_plans)
        return round(avg, 2)

    def _calculate_avg_subtask_length(self) -> float:
        """Calculate average subtask description length"""
        # Collect all subtask description lengths
        all_lengths = []
        for plan in self.parsed_plans.values():
            for subtask in plan.subtasks:
                all_lengths.append(len(subtask.description))

        if len(all_lengths) == 0:
            return 0.0

        # Calculate mean
        mean = sum(all_lengths) / len(all_lengths)
        return round(mean, 2)

    # ========== Dependency metrics ==========

    def _check_has_irrelevant_dependencies(self) -> float:
        """Check for irrelevant dependencies (LLM judgment, batch processing); returns the ratio of tasks with issues"""
        config = ConfigLoader()
        system_prompt = """You are a plan quality check expert. You need to check whether each plan contains subtasks that are obviously irrelevant to the original task.

Notes:
1. Only flag "obviously" irrelevant subtasks; if it is not particularly obvious, treat it as relevant
2. For each task-plan pair, determine whether all subtasks in the plan are related to the task
3. Return the list of task IDs that have irrelevant subtasks"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # Batch processing, 10 plans per batch
        batch_size = 10
        task_items = list(self.plans.items())
        total_tasks_with_issues = 0

        for i in range(0, len(task_items), batch_size):
            batch = task_items[i : i + batch_size]

            # Build batch check prompt
            batch_info = []
            for task_id, plan_text in batch:
                task_desc = self.tasks.get(task_id, "")
                batch_info.append(
                    {"task_id": task_id, "task": task_desc, "plan": plan_text}
                )

            user_prompt = f"""Please check the plans for the following {len(batch)} tasks and identify tasks that contain obviously irrelevant subtasks:

"""
            for idx, item in enumerate(batch_info, 1):
                user_prompt += f"""
Task ID: {item['task_id']}
Task content: {item['task']}
Task plan:
{item['plan']}

---
"""

            user_prompt += """
Please analyze each plan and identify task IDs that contain obviously irrelevant subtasks."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=IrrelevantDependenciesCheckResult,
                )

                # Count tasks with issues in this batch (only count valid task IDs)
                for task_id in result.task_ids_with_issues:
                    if task_id in self.tasks:
                        total_tasks_with_issues += 1

            except Exception as e:
                print(f"Batch check for irrelevant dependencies failed (batch {i // batch_size + 1}): {e}")
                continue

        # Calculate ratio
        total_tasks = len(self.plans)
        if total_tasks == 0:
            return 0.0
        return round((total_tasks_with_issues / total_tasks) * 100, 2)

    def _check_dependency_cycle_ratio(self) -> float:
        """Use topological sort (Kahn's algorithm) to detect dependency cycles in all plans; returns the ratio of tasks with cycles."""
        if len(self.parsed_plans) == 0:
            return 0.0

        total_tasks_with_cycles = 0

        for task_id, plan in self.parsed_plans.items():
            if len(plan.subtasks) == 0:
                continue

            # Build adjacency list and in-degree table
            indices = {st.index for st in plan.subtasks}
            in_degree = {idx: 0 for idx in indices}
            adj = {idx: [] for idx in indices}

            for st in plan.subtasks:
                for dep in st.dependencies:
                    if dep in indices:
                        adj[dep].append(st.index)
                        in_degree[st.index] += 1

            # Kahn's algorithm: topological sort
            queue = [idx for idx, deg in in_degree.items() if deg == 0]
            visited_count = 0

            while queue:
                node = queue.pop()
                visited_count += 1
                for neighbor in adj[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # If not all nodes are visited, a cycle exists
            if visited_count < len(indices):
                total_tasks_with_cycles += 1

        total_tasks = len(self.parsed_plans)
        return round((total_tasks_with_cycles / total_tasks) * 100, 2)

    def _check_dependency_cycle_ratio_llm(self) -> float:
        """Use LLM to detect dependency cycles in all plans; returns the ratio of tasks with cycles."""
        if len(self.plans) == 0:
            return 0.0

        config = ConfigLoader()
        system_prompt = """You are an expert in analyzing dependency graphs for multi-agent system plans. You need to check whether each plan's dependency graph contains cycles (loops).

Definition of a cycle: if, according to the plan, certain subtasks may be executed repeatedly, a cycle exists.

Please carefully analyze each plan and identify those with cycles."""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        batch_size = 10
        task_items = list(self.plans.items())
        total_tasks_with_cycles = 0

        for i in range(0, len(task_items), batch_size):
            batch = task_items[i : i + batch_size]

            user_prompt = f"""Please check the dependencies of the following {len(batch)} plans and identify those with cycles:

"""
            for task_id, plan_text in batch:
                user_prompt += f"""Task ID: {task_id}
Task plan:
{plan_text}

---
"""

            user_prompt += """
For each plan, analyze its dependency graph and determine whether a cycle exists. If a cycle exists, describe the cycle path."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=DependencyCycleCheckResult,
                )

                # Count tasks with cycles in this batch (only count valid task IDs)
                for task_id in result.task_ids_with_cycles:
                    if task_id in self.plans:
                        total_tasks_with_cycles += 1

            except Exception as e:
                print(f"Batch check for dependency cycles failed (batch {i // batch_size + 1}): {e}")
                continue

        total_tasks = len(self.plans)
        return round((total_tasks_with_cycles / total_tasks) * 100, 2)

    # ========== Agent capability match metrics ==========

    def _calculate_agent_task_mismatch_ratio(self) -> Dict[str, float]:
        """
        Calculate the ratio of subtasks mismatched to each Agent's capabilities (LLM judgment, batch processing).

        Returns:
            Dict[str, float]: Agent name -> mismatch ratio (0.0-100.0)
        """
        config = ConfigLoader()
        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
        )

        # 1. Collect all subtasks for each Agent across all plans
        agent_subtasks: Dict[str, List[str]] = {}  # Agent name -> list of subtask descriptions

        for plan in self.parsed_plans.values():
            for subtask in plan.subtasks:
                agent_name = subtask.agent_name
                if agent_name not in agent_subtasks:
                    agent_subtasks[agent_name] = []
                agent_subtasks[agent_name].append(subtask.description)

        # 2. Batch-evaluate subtasks for each Agent
        mismatch_ratios: Dict[str, float] = {}
        batch_size = 10  # 10 subtasks per batch

        for agent_name, subtasks in agent_subtasks.items():
            # Get this Agent's capability description
            agent_capability = self.members.get(agent_name, "")

            if not agent_capability:
                # If no capability description, default mismatch ratio to 0
                mismatch_ratios[agent_name] = 0.0
                continue

            # Each Agent uses a different system_prompt
            system_prompt = f"""You are an Agent capability matching expert. You need to evaluate whether the given subtasks obviously do not match the Agent's capability description.

Agent capability description:
{agent_capability}

Notes:
1. Only flag "obviously" mismatched subtasks; if uncertain or possibly matching, treat as matching
2. Analyze whether each subtask falls within this Agent's capability scope
3. Return the indices of mismatched subtasks (0-based)"""

            # Process this Agent's subtasks in batches
            all_unmatched_indices = []

            for i in range(0, len(subtasks), batch_size):
                batch = subtasks[i : i + batch_size]

                user_prompt = f"""The following are subtasks assigned to this Agent. Please identify those that obviously do not match its capabilities:

"""
                for idx, subtask_desc in enumerate(batch):
                    user_prompt += f"{idx}. {subtask_desc}\n\n"

                user_prompt += """
Please analyze each subtask and identify those that obviously do not match this Agent's capabilities."""

                try:
                    llm_client.reset(system_prompt=system_prompt)
                    result = llm_client.ask_structured(
                        user_input=[user_prompt],
                        model_cls=AgentCapabilityMatchResult,
                    )

                    # Convert batch-local indices to global indices (with bounds check)
                    for local_idx in result.unmatched_subtask_indices:
                        if 0 <= local_idx < len(batch):
                            all_unmatched_indices.append(i + local_idx)

                except Exception as e:
                    print(
                        f"Agent {agent_name} capability match check failed (batch {i // batch_size + 1}): {e}"
                    )
                    continue

            # Calculate mismatch ratio
            total = len(subtasks)
            unmatched_count = len(all_unmatched_indices)

            if total == 0:
                mismatch_ratios[agent_name] = 0.0
            else:
                ratio = (unmatched_count / total) * 100
                mismatch_ratios[agent_name] = round(ratio, 2)

        return mismatch_ratios

    def _calculate_agent_task_mismatch_ratio_2(self) -> Dict[str, float]:
        """
        Calculate the ratio of subtasks mismatched to each Agent's capabilities
        (LLM judgment based on full plan context, batch processing).

        Difference from _calculate_agent_task_mismatch_ratio:
        Provides the complete plan context (all subtasks and their dependencies) to the LLM,
        so it understands this is a multi-agent collaborative plan and avoids misclassifying
        subtasks that reference intermediate results from other Agents as "out of scope."

        Returns:
            Dict[str, float]: Agent name -> mismatch ratio (0.0-100.0)
        """
        config = ConfigLoader()

        # Build Agent capability description text
        members_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in self.members.items()
        )

        system_prompt = f"""You are a Plan quality evaluation expert for multi-agent systems.

The following are the capability descriptions of all Agents in the system:
{members_desc}

You need to analyze each complete Plan and determine which subtasks have been assigned to Agents that obviously do not match their capabilities.

Important background:
1. This is a multi-agent collaborative system; the Planner decomposes a task into multiple subtasks and assigns them to different Agents
2. Subtasks have dependencies ([dep:X] means it depends on the output of subtask X)
3. A subtask may say "based on the result from the previous Agent..." or "using the output of subtask X..."; this is normal collaboration and does not count as out of scope
4. When judging a match, only focus on whether what the subtask asks this Agent to "do itself" falls within its capability scope
5. Only flag "obviously" mismatched subtasks; if uncertain, treat as matching"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Submit plans in batches
        batch_size = 5
        plan_items = list(self.parsed_plans.items())

        # Collect (total subtasks, mismatched subtasks) for each Agent
        agent_total: Dict[str, int] = {}
        agent_unmatched: Dict[str, int] = {}

        # Count total subtasks per Agent first
        for plan in self.parsed_plans.values():
            for subtask in plan.subtasks:
                agent_name = subtask.agent_name
                agent_total[agent_name] = agent_total.get(agent_name, 0) + 1

        for i in range(0, len(plan_items), batch_size):
            batch = plan_items[i : i + batch_size]

            user_prompt = f"""The following are {len(batch)} complete Plans. Please analyze each Plan and identify subtasks assigned to Agents that obviously do not match their capabilities.

"""
            for task_id, plan in batch:
                user_prompt += f"=== Plan (task_id: {task_id}) ===\n"
                for subtask in plan.subtasks:
                    deps_str = (
                        f" [dep:{','.join(map(str, subtask.dependencies))}]"
                        if subtask.dependencies
                        else ""
                    )
                    user_prompt += f"{subtask.index}. {subtask.agent_name} {deps_str}: {subtask.description}\n"
                user_prompt += "\n"

            user_prompt += """Please analyze each Plan and identify subtasks that obviously do not match the corresponding Agent's capabilities.
For each problematic Plan, return its task_id and the mismatched subtask numbers (using the original subtask numbers from the plan).
If a Plan has no mismatched subtasks, it does not need to be included in the result.
Note: receiving results from other Agents as input is normal collaboration and does not count as out of scope."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=AgentCapabilityMatchResult2,
                )

                # Map task_id and subtask numbers back to Agents
                for item in result.task_mismatch_map:
                    task_id = item.task_id
                    if task_id not in self.parsed_plans:
                        continue
                    plan = self.parsed_plans[task_id]
                    for subtask_idx in item.unmatched_subtask_indices:
                        subtask = plan.get_subtask_by_index(subtask_idx)
                        if subtask is None:
                            continue
                        agent_name = subtask.agent_name
                        agent_unmatched[agent_name] = agent_unmatched.get(agent_name, 0) + 1

            except Exception as e:
                print(f"Agent capability match check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 2. Calculate mismatch ratio for each Agent
        mismatch_ratios: Dict[str, float] = {}
        for agent_name, total in agent_total.items():
            unmatched = agent_unmatched.get(agent_name, 0)
            if total == 0:
                mismatch_ratios[agent_name] = 0.0
            else:
                mismatch_ratios[agent_name] = round((unmatched / total) * 100, 2)

        return mismatch_ratios

    def _calculate_agent_participation_rate(self) -> Dict[str, float]:
        """
        Calculate Agent participation frequency (ratio of tasks in which each Agent appears).

        Returns:
            Dict[str, float]: Agent name -> participation rate (percentage)
        """
        if len(self.parsed_plans) == 0:
            return {}

        agent_task_count: Dict[str, int] = {}

        for plan in self.parsed_plans.values():
            agents_in_plan = set()
            for subtask in plan.subtasks:
                agents_in_plan.add(subtask.agent_name)

            for agent_name in agents_in_plan:
                if agent_name not in agent_task_count:
                    agent_task_count[agent_name] = 0
                agent_task_count[agent_name] += 1

        total_tasks = len(self.parsed_plans)
        result: Dict[str, float] = {}
        for agent_name, count in agent_task_count.items():
            participation_rate = (count / total_tasks) * 100
            result[agent_name] = round(participation_rate, 2)

        return result

    # ========== Subtask description quality metrics ==========

    def _check_subtasks_require_excessive_effort_ratio(self) -> float:
        """
        Check whether subtasks require excessive reasoning/tool invocation/output (LLM judgment, batch processing).

        Returns:
            float: Ratio of subtasks requiring excessive effort (0.0-100.0)
        """
        config = ConfigLoader()
        system_prompt = """You are a task analysis expert. You need to determine whether the given subtask description requires excessive reasoning, tool invocations, or output from the executing Agent.

Notes:
1. Only flag subtasks that obviously require excessive thinking, invocations, or output
2. If a subtask has a clear scope and boundary, it is not considered excessive
3. Return the indices of subtasks requiring excessive effort (0-based)"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Collect all subtasks
        all_subtasks = []  # stores (task_id, subtask) tuples

        for task_id, plan in self.parsed_plans.items():
            for subtask in plan.subtasks:
                all_subtasks.append((task_id, subtask))

        if len(all_subtasks) == 0:
            return 0.0

        # 2. Use LLM in batches to judge which subtasks require excessive effort
        batch_size = 10
        all_excessive_effort_indices = []

        for i in range(0, len(all_subtasks), batch_size):
            batch = all_subtasks[i : i + batch_size]

            user_prompt = f"""The following are {len(batch)} subtask descriptions. Please identify those that require excessive reasoning/invocations/output:

"""
            for idx, (_, subtask) in enumerate(batch):
                user_prompt += (
                    f"{idx}. [{subtask.agent_name}] {subtask.description}\n\n"
                )

            user_prompt += """
Please carefully analyze each subtask and determine whether it requires excessive reasoning, invocations, or output from the executor.
Return the list of subtask indices that require excessive effort."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=ExcessiveEffortCheckResult,
                )

                # Convert batch-local indices to global indices (with bounds check)
                for local_idx in result.excessive_effort_indices:
                    if 0 <= local_idx < len(batch):
                        all_excessive_effort_indices.append(i + local_idx)

            except Exception as e:
                print(f"Excessive effort check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 3. Calculate ratio
        total_subtasks = len(all_subtasks)
        excessive_count = len(all_excessive_effort_indices)

        if total_subtasks == 0:
            return 0.0

        ratio = (excessive_count / total_subtasks) * 100
        return round(ratio, 2)

    def _calculate_dependent_subtasks_no_clear_middle_result_ratio(self) -> float:
        """
        Calculate the ratio of depended-upon subtasks that do not specify a clear intermediate output.

        Returns:
            float: Ratio of depended-upon subtasks without a clear output (0.0-100.0)
        """
        config = ConfigLoader()
        system_prompt = """You are a task analysis expert. You need to determine whether the given subtask description explicitly specifies the intermediate result to be output.

Notes:
1. If the task description explicitly mentions output, saving, returning, or generating a result, it counts as having a clear output
2. If it only describes what to do without stating what result to output, it counts as no clear output
3. Return the indices of subtasks that do not clearly specify an output (0-based)"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Collect all subtasks that are depended upon by other subtasks
        dependent_subtasks = []  # stores (plan_id, subtask) tuples

        for task_id, plan in self.parsed_plans.items():
            # Find all depended-upon subtask indices
            dependent_indices = set()
            for subtask in plan.subtasks:
                for dep_idx in subtask.dependencies:
                    dependent_indices.add(dep_idx)

            # Collect depended-upon subtasks
            for subtask in plan.subtasks:
                if subtask.index in dependent_indices:
                    dependent_subtasks.append((task_id, subtask))

        if len(dependent_subtasks) == 0:
            return 0.0

        # 2. Use LLM in batches to judge which ones do not clearly specify an output
        batch_size = 10
        all_no_clear_output_indices = []

        for i in range(0, len(dependent_subtasks), batch_size):
            batch = dependent_subtasks[i : i + batch_size]

            user_prompt = f"""The following are subtasks that other subtasks depend on. Please identify those that do not clearly specify an intermediate output result:

"""
            for idx, (task_id, subtask) in enumerate(batch):
                user_prompt += f"{idx}. {subtask.description}\n\n"

            user_prompt += """
Please analyze each subtask and identify those that do not clearly specify an intermediate output result."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=DependentSubtaskOutputCheckResult,
                )

                # Convert batch-local indices to global indices (with bounds check)
                for local_idx in result.no_clear_output_indices:
                    if 0 <= local_idx < len(batch):
                        all_no_clear_output_indices.append(i + local_idx)

            except Exception as e:
                print(f"Dependent subtask output check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 3. Calculate ratio
        total_dependent_subtasks = len(dependent_subtasks)
        no_clear_output_count = len(all_no_clear_output_indices)

        if total_dependent_subtasks == 0:
            return 0.0

        ratio = (no_clear_output_count / total_dependent_subtasks) * 100
        return round(ratio, 2)

    def _check_subtasks_not_clearly_specify_actions_ratio(self) -> float:
        """
        Check whether subtasks clearly state what to do and to what extent (LLM judgment, batch processing).

        Returns:
            float: Ratio of subtasks that are unclear (0.0-100.0)
        """
        config = ConfigLoader()
        system_prompt = """You are a task analysis expert. You need to determine whether the given subtask description clearly states what to do, such as:
1. What to do (specific action/operation)
2. To what extent (completion criteria/expected result)

Notes:
1. A clear subtask should let the executor know exactly what operation to perform and what counts as done
2. If the subtask description is too vague or general, or only states the goal without specifying the action, it counts as unclear
3. If the subtask only states the action without a completion criterion, it also counts as unclear
4. Return the indices of unclear subtasks (0-based)
"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Collect all subtasks
        all_subtasks = []  # stores (task_id, subtask) tuples

        for task_id, plan in self.parsed_plans.items():
            for subtask in plan.subtasks:
                all_subtasks.append((task_id, subtask))

        if len(all_subtasks) == 0:
            return 0.0

        # 2. Use LLM in batches to judge which subtasks do not clearly state what to do
        batch_size = 10
        all_unclear_indices = []

        for i in range(0, len(all_subtasks), batch_size):
            batch = all_subtasks[i : i + batch_size]

            user_prompt = f"""The following are {len(batch)} subtask descriptions. Please identify those that do not clearly state what to do or to what extent:

"""
            for idx, (task_id, subtask) in enumerate(batch):
                user_prompt += (
                    f"{idx}. [{subtask.agent_name}] {subtask.description}\n\n"
                )

            user_prompt += """
Please carefully analyze each subtask and determine whether it clearly states what to do, such as:
1. Specific operation to perform
2. Completion criteria/extent

Return the list of unclear subtask indices."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=SubtaskActionClarityCheckResult,
                )

                # Convert batch-local indices to global indices (with bounds check)
                for local_idx in result.unclear_subtask_indices:
                    if 0 <= local_idx < len(batch):
                        all_unclear_indices.append(i + local_idx)

            except Exception as e:
                print(f"Subtask action clarity check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 3. Calculate ratio
        total_subtasks = len(all_subtasks)
        unclear_count = len(all_unclear_indices)

        if total_subtasks == 0:
            return 0.0

        ratio = (unclear_count / total_subtasks) * 100
        return round(ratio, 2)

    def _check_has_verification_intent_ratio(self) -> float:
        """
        Check whether subtask descriptions require verification/checking before outputting results (LLM judgment, batch processing).

        Returns:
            float: Ratio of subtasks requiring verification before output (0.0-100.0)
        """
        config = ConfigLoader()
        system_prompt = """You are a task analysis expert. You need to determine whether the given subtask description explicitly requires the executor to verify/check results after completing the main operation, before outputting.

Notes:
1. "Verification intent" means the subtask description explicitly requires the executor to verify/check the correctness of results after completing the main operation, before outputting
2. Return the indices of subtasks with verification intent (0-based)"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Collect all subtasks
        all_subtasks = []  # stores (task_id, subtask) tuples

        for task_id, plan in self.parsed_plans.items():
            for subtask in plan.subtasks:
                all_subtasks.append((task_id, subtask))

        if len(all_subtasks) == 0:
            return 0.0

        # 2. Use LLM in batches to judge which subtasks require verification before output
        batch_size = 10
        all_verification_intent_indices = []

        for i in range(0, len(all_subtasks), batch_size):
            batch = all_subtasks[i : i + batch_size]

            user_prompt = f"""The following are {len(batch)} subtask descriptions. Please identify those that require verification/checking before outputting results:

"""
            for idx, (task_id, subtask) in enumerate(batch):
                user_prompt += (
                    f"{idx}. [{subtask.agent_name}] {subtask.description}\n\n"
                )

            user_prompt += """
Please carefully analyze each subtask and determine whether it explicitly requires verifying/checking results before outputting.
Return the list of subtask indices with verification intent."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=VerificationIntentCheckResult,
                )

                # Convert batch-local indices to global indices (with bounds check)
                for local_idx in result.has_verification_intent_indices:
                    if 0 <= local_idx < len(batch):
                        all_verification_intent_indices.append(i + local_idx)

            except Exception as e:
                print(f"Subtask verification intent check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 3. Calculate ratio
        total_subtasks = len(all_subtasks)
        verification_intent_count = len(all_verification_intent_indices)

        if total_subtasks == 0:
            return 0.0

        ratio = (verification_intent_count / total_subtasks) * 100
        return round(ratio, 2)

    def _check_has_dedicated_verification_tasks_ratio(self) -> float:
        """
        Check whether each plan contains subtasks dedicated to verifying previous subtasks (LLM judgment, batch processing).

        Returns:
            float: Ratio of dedicated verification subtasks to all subtasks (0.0-100.0)
        """
        config = ConfigLoader()
        system_prompt = """You are a task analysis expert. You need to determine whether a given plan contains subtasks whose primary purpose is to verify/check/confirm the results of previous subtasks.

Notes:
1. A "dedicated verification task" is a subtask whose main purpose is to verify/check/confirm whether the result of a previous subtask is correct
2. It should depend on the subtask being verified (in dependencies)
3. For plans that have dedicated verification subtasks, return the task_id and the list of verification subtask numbers
4. If a plan has no dedicated verification tasks, it does not need to be included in the result"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        # 1. Collect all plans (full plan context is needed to judge whether a subtask is a dedicated verification task)
        all_plans = []  # stores (task_id, plan) tuples

        for task_id, plan in self.parsed_plans.items():
            if len(plan.subtasks) > 0:
                all_plans.append((task_id, plan))

        if len(all_plans) == 0:
            return 0.0

        # Count total subtasks and dedicated verification task count
        total_subtasks = sum(len(plan.subtasks) for _, plan in all_plans)
        total_dedicated_verification_tasks = 0

        # 2. Process plans in batches (each plan needs full context)
        batch_size = 5  # 5 plans per batch since each may have multiple subtasks

        for i in range(0, len(all_plans), batch_size):
            batch = all_plans[i : i + batch_size]

            user_prompt = f"""The following are {len(batch)} plans. Please identify subtasks in each plan that are dedicated to verifying previous subtasks:

"""
            for _, (task_id, plan) in enumerate(batch):
                user_prompt += f"=== task_id: {task_id} ===\n"
                for subtask in plan.subtasks:
                    deps_str = (
                        f" [dep:{','.join(map(str, subtask.dependencies))}]"
                        if subtask.dependencies
                        else ""
                    )
                    user_prompt += f"{subtask.index}. {subtask.agent_name} {deps_str}: {subtask.description}\n"
                user_prompt += "\n"

            user_prompt += """
Please analyze each plan and identify subtasks dedicated to verifying previous subtasks.
Return a list where each element contains a task_id and the list of verification subtask numbers in that plan (using original subtask numbers from the plan, e.g., 1, 2, 3...).
If a plan has no dedicated verification tasks, do not include it in the result."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=DedicatedVerificationTaskCheckResult,
                )

                # Count dedicated verification tasks in this batch
                for item in result.task_verification_map:
                    # Verify the task_id is in the current batch
                    if item.task_id in self.parsed_plans:
                        total_dedicated_verification_tasks += len(item.verification_indices)

            except Exception as e:
                print(f"Dedicated verification task check failed (batch {i // batch_size + 1}): {e}")
                continue

        # 3. Calculate ratio
        if total_subtasks == 0:
            return 0.0

        ratio = (total_dedicated_verification_tasks / total_subtasks) * 100
        return round(ratio, 2)

    # ========== Prior assumption injection metrics ==========

    def _build_plan_text(self, plan: "Plan") -> str:
        """Convert a Plan object to LLM-readable text format"""
        lines = []
        for subtask in plan.subtasks:
            deps_str = (
                f" [dep:{','.join(map(str, subtask.dependencies))}]"
                if subtask.dependencies
                else ""
            )
            lines.append(f"{subtask.index}. {subtask.agent_name}{deps_str}: {subtask.description}")
        return "\n".join(lines)

    def _check_assumption_injection_ratio(self) -> float:
        """
        Check whether subtasks contain prior assumptions injected by the Planner
        (LLM judgment, per task-plan pair).

        Prior assumption injection: the Planner writes into a subtask information not explicitly
        given in the user's task as a premise, such as expanding the query scope, redefining the
        meaning of the user's wording, or replacing specific references with broad category searches.

        Returns:
            float: Ratio of subtasks with injected prior assumptions (percentage)
        """
        config = ConfigLoader()
        system_prompt = """You are a Plan quality evaluation expert for multi-agent systems, specializing in detecting whether the Planner injected prior assumptions into subtasks.

Definition of prior assumption injection: when assigning subtasks, the Planner writes information not explicitly given in the user's original request as a premise into the subtask description. Typical manifestations include:
- Expanding the scope of the user's query (e.g., changing "find one specific X" to "search all X")
- Redefining the meaning of the user's wording (e.g., interpreting a specific reference as a broad category concept)
- Replacing the user's specific singular reference with a plural or global search

Notes:
1. Only flag "obvious" prior assumption injections; do not flag uncertain cases
2. For each flagged subtask, point out the specific wording in the original task and what assumption the subtask makes"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        total_subtask_count = 0
        total_flagged_count = 0

        for task_id, plan in self.parsed_plans.items():
            if len(plan.subtasks) == 0:
                continue

            total_subtask_count += len(plan.subtasks)
            task_text = self.tasks.get(task_id, "")
            if not task_text:
                continue

            plan_text = self._build_plan_text(plan)
            user_prompt = f"""Please analyze the following task and plan, and identify subtasks in the plan where the Planner injected prior assumptions.

Original user task:
{task_text}

Plan generated by Planner:
{plan_text}

Please compare the original task text with each subtask description one by one, and identify which subtask content depends on assumptions not explicitly given in the task.
Use the original subtask numbers from the plan, and in the reason point out the specific discrepancy between the task wording and the subtask assumption."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=AssumptionInjectionCheckResult,
                )
                valid_indices = {st.index for st in plan.subtasks}
                for item in result.flagged_subtasks:
                    if item.subtask_index in valid_indices:
                        total_flagged_count += 1
            except Exception as e:
                print(f"Prior assumption injection check failed (task_id: {task_id}): {e}")
                continue

        if total_subtask_count == 0:
            return 0.0
        return round((total_flagged_count / total_subtask_count) * 100, 2)

    def _check_vague_concretization_ratio(self) -> float:
        """
        Check whether subtasks over-concretize vague user expressions
        (LLM judgment, per task-plan pair).

        Vague expression over-concretization: the user's task contains vague or open-ended
        expressions, and the Planner prematurely resolves the ambiguity by locking it into a
        specific interpretation that is not the only reasonable one.

        Returns:
            float: Ratio of subtasks with over-concretization (percentage)
        """
        config = ConfigLoader()
        system_prompt = """You are a Plan quality evaluation expert for multi-agent systems, specializing in detecting whether the Planner prematurely resolved ambiguity in user expressions.

Definition of vague expression over-concretization: a certain expression in the user's task is inherently vague, open-ended, or has multiple reasonable interpretations, but the Planner unilaterally chose one specific interpretation and directly wrote it into execution steps, without preserving the uncertainty or first verifying its interpretation.

Notes:
1. Only flag "obvious" over-concretizations; do not flag uncertain cases
2. For each flagged subtask, point out which expression in the task is vague and what specific interpretation the Planner chose"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        total_subtask_count = 0
        total_flagged_count = 0

        for task_id, plan in self.parsed_plans.items():
            if len(plan.subtasks) == 0:
                continue

            total_subtask_count += len(plan.subtasks)
            task_text = self.tasks.get(task_id, "")
            if not task_text:
                continue

            plan_text = self._build_plan_text(plan)
            user_prompt = f"""Please analyze the following task and plan, and identify subtasks where the Planner prematurely resolved ambiguity in user expressions.

Original user task:
{task_text}

Plan generated by Planner:
{plan_text}

Please analyze each subtask individually to determine whether it arbitrarily concretized a vague or open-ended expression in the task into a specific interpretation.
Use the original subtask numbers from the plan, and in the reason point out which expression in the task is vague and how the Planner locked it into a specific direction."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=VagueConcretizationCheckResult,
                )
                valid_indices = {st.index for st in plan.subtasks}
                for item in result.flagged_subtasks:
                    if item.subtask_index in valid_indices:
                        total_flagged_count += 1
            except Exception as e:
                print(f"Vague expression over-concretization check failed (task_id: {task_id}): {e}")
                continue

        if total_subtask_count == 0:
            return 0.0
        return round((total_flagged_count / total_subtask_count) * 100, 2)

    def _check_premature_commitment_ratio(self) -> float:
        """
        Check whether the plan contains premature path commitments
        (LLM judgment, per task-plan pair).

        Premature path commitment: without sufficient evidence, the Planner skips the step of
        'first confirming semantic anchors' and directly writes an unverified interpretation as
        fact into execution steps, instead of treating the assumption as something to be verified.
        A normal plan should first have a step to "confirm task semantics / find the correct referent"
        before proceeding to execution.

        Returns:
            float: Ratio of subtasks reflecting premature path commitment (percentage)
        """
        config = ConfigLoader()
        system_prompt = """You are a Plan quality evaluation expert for multi-agent systems, specializing in detecting whether the Planner prematurely committed to an interpretation path without sufficient evidence.

Definition of premature path commitment: the Planner skips the step of "confirming semantic anchors" (e.g., first searching, first locating, first understanding context), and directly writes a certain interpretation of the task as known fact into execution steps. The assumption is treated as truth rather than as a hypothesis to be verified.

Judgment method:
- Check whether the plan skips the step of "first confirming the actual referent of the problem"
- Check whether any subtask directly executes a specific interpretation without a preceding confirmation step in the plan

Notes:
1. This is a plan-level judgment that requires considering the overall structure of the plan, not just individual subtasks
2. Flag subtasks that "directly execute an unverified interpretation"
3. Only flag "obvious" premature commitments; do not flag uncertain cases
4. In the reason, state: what confirmation step was skipped in the plan, and what unverified interpretation path the subtask commits to"""

        llm_client = LLMClient(
            api_key=config.get("llm.api_key"),
            base_url=config.get("llm.base_url"),
            model=config.get("llm.model"),
            temperature=0.0,
            max_tokens=config.get("llm.max_tokens"),
            system_prompt=system_prompt,
        )

        total_subtask_count = 0
        total_flagged_count = 0

        for task_id, plan in self.parsed_plans.items():
            if len(plan.subtasks) == 0:
                continue

            total_subtask_count += len(plan.subtasks)
            task_text = self.tasks.get(task_id, "")
            if not task_text:
                continue

            plan_text = self._build_plan_text(plan)
            user_prompt = f"""Please analyze the following task and plan, and identify subtasks where the Planner prematurely committed to an interpretation path without sufficient evidence.

Original user task:
{task_text}

Plan generated by Planner:
{plan_text}

Please analyze the overall structure of the plan:
1. Does the plan have a step to "confirm semantic anchors" (locating, understanding context, confirming the actual referent)?
2. Which subtasks directly execute a certain interpretation without that interpretation being previously verified in the plan?

Use the original subtask numbers from the plan, and in the reason state what confirmation step the subtask skipped and what unverified interpretation path it commits to."""

            try:
                llm_client.reset(system_prompt=system_prompt)
                result = llm_client.ask_structured(
                    user_input=[user_prompt],
                    model_cls=PrematureCommitmentCheckResult,
                )
                valid_indices = {st.index for st in plan.subtasks}
                for item in result.flagged_subtasks:
                    if item.subtask_index in valid_indices:
                        total_flagged_count += 1
            except Exception as e:
                print(f"Premature path commitment check failed (task_id: {task_id}): {e}")
                continue

        if total_subtask_count == 0:
            return 0.0
        return round((total_flagged_count / total_subtask_count) * 100, 2)

    # ========== Metric calculation entry point ==========

    def calculate_metric(self, metric_type: PlanMetricType) -> Dict[str, Any]:
        """
        Call the corresponding calculation function based on metric type.

        Args:
            metric_type: PlanMetricType enum value

        Returns:
            Dict[str, Any]: Dictionary containing "metric" (metric name) and "result" (calculated value)
        """
        # Mapping from metric type to calculation function
        metric_func_map = {
            # Structure-related metrics
            PlanMetricType.AVG_SUBTASKS_PER_TASK: self._calculate_avg_subtasks_per_task,
            PlanMetricType.AVG_SUBTASK_LENGTH: self._calculate_avg_subtask_length,
            # Dependency metrics
            PlanMetricType.HAS_IRRELEVANT_DEPENDENCIES: self._check_has_irrelevant_dependencies,
            PlanMetricType.DEPENDENCY_CYCLE_RATIO: self._check_dependency_cycle_ratio,
            PlanMetricType.DEPENDENCY_CYCLE_RATIO_LLM: self._check_dependency_cycle_ratio_llm,
            # Agent capability match metrics
            PlanMetricType.AGENT_TASK_MISMATCH_RATIO: self._calculate_agent_task_mismatch_ratio_2,
            PlanMetricType.AGENT_PARTICIPATION_RATE: self._calculate_agent_participation_rate,
            # Subtask description quality metrics
            PlanMetricType.SUBTASKS_REQUIRE_EXCESSIVE_EFFORT_RATIO: self._check_subtasks_require_excessive_effort_ratio,
            PlanMetricType.DEPENDENT_SUBTASKS_NO_CLEAR_MIDDLE_RESULT_RATIO: self._calculate_dependent_subtasks_no_clear_middle_result_ratio,
            PlanMetricType.SUBTASKS_NO_CLEARLY_SPECIFY_ACTIONS_RATIO: self._check_subtasks_not_clearly_specify_actions_ratio,
            PlanMetricType.VERIFICATION_INTENT_RATIO: self._check_has_verification_intent_ratio,
            PlanMetricType.DEDICATED_VERIFICATION_TASKS_RATIO: self._check_has_dedicated_verification_tasks_ratio,
            # Prior assumption injection metrics
            PlanMetricType.ASSUMPTION_INJECTION_RATIO: self._check_assumption_injection_ratio,
            PlanMetricType.VAGUE_CONCRETIZATION_RATIO: self._check_vague_concretization_ratio,
            PlanMetricType.PREMATURE_COMMITMENT_RATIO: self._check_premature_commitment_ratio,
        }

        if metric_type not in metric_func_map:
            raise ValueError(f"Unknown metric type: {metric_type}")

        func = metric_func_map[metric_type]
        result = func()

        return {metric_type.value: result}

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all metrics.

        Returns:
            Dict[str, Any]: Calculation results for all metrics; key is metric name, value is result
        """
        results = {}
        for metric_type in PlanMetricType:
            try:
                metric_result = self.calculate_metric(metric_type)
                metric_name = metric_type.value
                results[metric_name] = metric_result[metric_name]
            except Exception as e:
                print(f"Failed to calculate metric {metric_type.value}: {e}")
                results[metric_type.value] = None
        return results


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all metrics and return results (without saving to file).

    Args:
        data: Dictionary containing members, plans, and tasks
            - members: Dict[str, str] - Agent name -> capability description
            - plans: Dict[str, str] - task ID -> plan text
            - tasks: Dict[str, str] - task ID -> task description

    Returns:
        Dict[str, Any]: Calculation results for all metrics, in the format:
            {
                "metric_name": {
                    "description": "metric description",
                    "value": calculated value
                },
                ...
            }
    """
    # Create calculator instance
    calculator = PlanMetricsCalculator(data)

    # Get all metrics to calculate
    all_metrics = list(PlanMetricType)
    total_metrics = len(all_metrics)

    results: Dict[str, Any] = {}

    for idx, metric_type in enumerate(all_metrics, 1):
        metric_name = metric_type.value

        print(f"[{idx}/{total_metrics}] Calculating metric: {metric_name}")
        metric_start_time = time.time()

        try:
            # Calculate metric
            metric_result = calculator.calculate_metric(metric_type)
            result_value = metric_result[metric_name]

            # Use new format: metric name -> {description, value}
            results[metric_name] = {
                "description": PLAN_METRIC_DESCRIPTIONS.get(metric_name, ""),
                "value": result_value,
            }

            metric_duration = round(time.time() - metric_start_time, 2)
            print(f"  -> Result: {result_value}, time: {metric_duration}s")

        except Exception as e:
            print(f"  -> Calculation failed: {e}")
            results[metric_name] = {
                "description": PLAN_METRIC_DESCRIPTIONS.get(metric_name, ""),
                "value": None,
            }

    print(f"\nAll metrics calculated!")
    return results


def calculate_and_save_metrics(
    input_file_path: str, output_file_name: str = "metrics_result.json"
) -> Dict[str, Any]:
    """
    Load the specified JSON file, calculate all metrics, and save to a result file in the same directory.

    Saves after each metric is calculated to prevent data loss on crash.
    Supports resuming: if the result file already exists and some metrics are already calculated, those are skipped.

    Args:
        input_file_path: Input file path, e.g. "tmp/logs/plan_test/get_plans_result.json"
        output_file_name: Output file name, default "metrics_result.json", saved in the same directory as the input

    Returns:
        Dict[str, Any]: Calculation results for all metrics
    """
    input_path = Path(input_file_path)

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file_path}")

    # Determine output file path (same directory)
    output_path = input_path.parent / output_file_name

    # Load input data
    print(f"Loading input file: {input_file_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create calculator instance
    calculator = PlanMetricsCalculator(data)
    print(f"Loaded {len(calculator.plans)} plans")

    # Load existing results (if any)
    results: Dict[str, Any] = {}
    if output_path.exists():
        print(f"Found existing result file: {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            # Count already-calculated metrics
            computed_count = sum(
                1 for v in results.values()
                if isinstance(v, dict) and v.get("value") is not None
            )
            print(f"Loaded {computed_count} already-calculated metrics")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Failed to load existing result file: {e}, recalculating all metrics")
            results = {}

    # Get all metrics to calculate
    all_metrics = list(PlanMetricType)
    total_metrics = len(all_metrics)

    for idx, metric_type in enumerate(all_metrics, 1):
        metric_name = metric_type.value

        # Check if already calculated
        if (
            metric_name in results
            and isinstance(results[metric_name], dict)
            and results[metric_name].get("value") is not None
        ):
            print(f"[{idx}/{total_metrics}] Skipping already-calculated metric: {metric_name}")
            continue

        print(f"[{idx}/{total_metrics}] Calculating metric: {metric_name}")
        metric_start_time = time.time()

        try:
            # Calculate metric
            metric_result = calculator.calculate_metric(metric_type)
            result_value = metric_result[metric_name]

            # Save in new format: metric name -> {description, value}
            results[metric_name] = {
                "description": PLAN_METRIC_DESCRIPTIONS.get(metric_name, ""),
                "value": result_value,
            }

            metric_duration = round(time.time() - metric_start_time, 2)
            print(f"  -> Result: {result_value}, time: {metric_duration}s")

        except Exception as e:
            print(f"  -> Calculation failed: {e}")
            results[metric_name] = {
                "description": PLAN_METRIC_DESCRIPTIONS.get(metric_name, ""),
                "value": None,
            }

        # Save result
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved to: {output_path}")

    print(f"\nAll metrics calculated! Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    input_file_path = project_root / "tmp/logs/plan_test/get_plans_result.json"

    calculate_and_save_metrics(input_file_path)
