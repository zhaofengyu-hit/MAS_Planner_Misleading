from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from ..log_parser import MetricType


class PlanMetricType(str, Enum):
    """Plan evaluation metric types"""

    # ========== Structure-related metrics ==========
    AVG_SUBTASKS_PER_TASK = "avg_subtasks_per_task"
    AVG_SUBTASK_LENGTH = "avg_subtask_length"

    # ========== Dependency metrics ==========
    HAS_IRRELEVANT_DEPENDENCIES = "has_irrelevant_dependencies"
    DEPENDENCY_CYCLE_RATIO = "dependency_cycle_ratio"
    DEPENDENCY_CYCLE_RATIO_LLM = "dependency_cycle_ratio_llm"

    # ========== Agent capability match metrics ==========
    AGENT_TASK_MISMATCH_RATIO = "agent_task_mismatch_ratio"
    AGENT_PARTICIPATION_RATE = "agent_participation_rate"

    # ========== Subtask description quality metrics ==========
    SUBTASKS_REQUIRE_EXCESSIVE_EFFORT_RATIO = "subtasks_require_excessive_effort_ratio"
    DEPENDENT_SUBTASKS_NO_CLEAR_MIDDLE_RESULT_RATIO = "dependent_subtasks_no_clear_middle_result_ratio"
    SUBTASKS_NO_CLEARLY_SPECIFY_ACTIONS_RATIO = "subtasks_no_clearly_specify_actions_ratio"
    VERIFICATION_INTENT_RATIO = "verification_intent_ratio"
    DEDICATED_VERIFICATION_TASKS_RATIO = "dedicated_verification_tasks_ratio"

    # ========== Prior assumption injection metrics ==========
    ASSUMPTION_INJECTION_RATIO = "assumption_injection_ratio"
    VAGUE_CONCRETIZATION_RATIO = "vague_concretization_ratio"
    PREMATURE_COMMITMENT_RATIO = "premature_commitment_ratio"

class MetricsAnalysis(BaseModel):
    metrics: List[MetricType] = Field(
        description="List of metric names being analyzed; must be chosen from the predefined metric enum"
    )
    has_improvement: bool = Field(description="Whether the child is closer to the fuzzing target than the parent")
    reason: str = Field(description="Explanation for the improvement judgment, referencing specific metric changes")


class QualityCheckResult(BaseModel):
    """Quality check result"""
    passed: bool = Field(description="Whether the quality check passed; True means passed, False means failed")
    violated_rules: List[str] = Field(
        description="List of violated rules; empty if passed is True; lists the full description of every violated rule if passed is False",
        default_factory=list
    )
    reason: str = Field(
        description="If the check failed, explains which rules were violated and how"
    )


class SeedMutation(BaseModel):
    """Seed mutation result"""
    new_description: str = Field(
        description="New Agent description generated after mutation; should improve upon the baseline while referencing positive and negative examples"
    )
    mutation_strategy: List[str] = Field(
        description="Explanation of the mutation strategies used, e.g. 'reinforce successful patterns from positive examples', 'avoid failure patterns from negative examples'"
    )
    key_changes: List[str] = Field(
        description="Key changes relative to the baseline, each described in one short sentence",
        default_factory=list
    )
    expected_improvement: str = Field(
        description="Expected improvement this mutation will bring, based on analysis of positive and negative examples"
    )


class SeedComparison(BaseModel):
    """Seed comparison result"""
    worst_seed_id: str = Field(
        description="The seed ID with the worst (most mediocre) performance among all compared seeds"
    )
    reason: str = Field(
        description="Reason for judging this seed as the worst, combining test results and metric analysis"
    )


class SinglePlanAnalysis(BaseModel):
    """Evaluation result for a single plan"""
    is_qualified: bool = Field(
        description="Whether this plan is qualified (able to meet the optimization target requirements)"
    )
    reason: str = Field(
        description="Detailed reasoning for the plan qualification judgment, combining metric changes and target requirements"
    )
    key_observations: List[str] = Field(
        description="Key observations about the plan, listing its strengths or weaknesses",
        default_factory=list
    )


class SubtaskData(BaseModel):
    """Subtask data"""
    index: int = Field(description="Subtask index number")
    agent_name: str = Field(description="Name of the responsible Agent (extracted as-is from the plan, do not modify)")
    description: str = Field(description="Task description (extracted as-is from the plan, preserve all details, do not modify)")
    dependencies: List[int] = Field(
        description="List of indices of other subtasks this one depends on; empty list if no dependencies",
        default_factory=list
    )


class PlanParseResult(BaseModel):
    """Plan parse result"""
    subtasks: List[SubtaskData] = Field(
        description="List of all subtasks in the order they appear in the plan",
        default_factory=list
    )


class TaskExplanation(BaseModel):
    """Task explanation entry"""
    task_id: str = Field(description="Task ID")
    explanation: str = Field(description="Explanation")


class IrrelevantDependenciesCheckResult(BaseModel):
    """Irrelevant dependency check result (batch)"""
    task_ids_with_issues: List[str] = Field(
        description="List of task IDs that contain obviously irrelevant subtasks",
        default_factory=list
    )
    explanations: List[TaskExplanation] = Field(
        description="For each problematic task ID, explains which subtasks are irrelevant to the original task and why",
        default_factory=list
    )


class TaskCycleDetail(BaseModel):
    """Task cycle detail entry"""
    task_id: str = Field(description="Task ID")
    cycle_path: str = Field(description="Specific cycle path (e.g. '1->2->3->1')")


class DependencyCycleCheckResult(BaseModel):
    """Dependency cycle check result (batch)"""
    task_ids_with_cycles: List[str] = Field(
        description="List of task IDs that contain dependency cycles",
        default_factory=list
    )
    cycle_details: List[TaskCycleDetail] = Field(
        description="For each task ID with a cycle, describes the specific cycle path",
        default_factory=list
    )


class IndexReason(BaseModel):
    """Index-to-reason mapping entry"""
    index: int = Field(description="Subtask index (0-based)")
    reason: str = Field(description="Reason explanation")


class AgentCapabilityMatchResult(BaseModel):
    """Agent capability match check result"""
    unmatched_subtask_indices: List[int] = Field(
        description="List of subtask indices (0-based, from the provided subtask list) that clearly do not match this Agent's capabilities",
        default_factory=list
    )
    reasons: List[IndexReason] = Field(
        description="For each unmatched subtask index, explains why it does not match",
        default_factory=list
    )


class TaskMismatchIndices(BaseModel):
    """Subtask indices in a single plan that do not match the Agent's capabilities"""
    task_id: str = Field(description="Task ID")
    unmatched_subtask_indices: List[int] = Field(
        description="List of subtask numbers in this plan that clearly do not match the corresponding Agent's capabilities (using the original subtask numbers from the plan)",
        default_factory=list
    )


class AgentCapabilityMatchResult2(BaseModel):
    """Agent capability match check result (batch processing of multiple plans based on full plan context)"""
    task_mismatch_map: List[TaskMismatchIndices] = Field(
        description="List of plans that have mismatched subtasks, each element containing task_id and mismatched subtask numbers",
        default_factory=list
    )
    reasons: List[TaskExplanation] = Field(
        description="For each task_id with mismatched subtasks, explains which subtasks do not match and why",
        default_factory=list
    )


class DependentSubtaskOutputCheckResult(BaseModel):
    """Dependent subtask output check result"""
    no_clear_output_indices: List[int] = Field(
        description="List of subtask indices (0-based) that do not clearly specify an intermediate output result",
        default_factory=list
    )
    reasons: List[IndexReason] = Field(
        description="For each subtask index without a clear output, explains why it is judged to have no specified output",
        default_factory=list
    )


class SubtaskActionClarityCheckResult(BaseModel):
    """Subtask action clarity check result"""
    unclear_subtask_indices: List[int] = Field(
        description="List of subtask indices (0-based) that do not clearly state what to do or to what extent",
        default_factory=list
    )
    reasons: List[IndexReason] = Field(
        description="For each unclear subtask index, explains why it is judged to be unclear",
        default_factory=list
    )


class VerificationIntentCheckResult(BaseModel):
    """Subtask verification intent check result"""
    has_verification_intent_indices: List[int] = Field(
        description="List of subtask indices (0-based) that require verification or checking before output after completion",
        default_factory=list
    )
    reasons: List[IndexReason] = Field(
        description="For each subtask index with verification intent, explains what the verification requirement is",
        default_factory=list
    )


class TaskVerificationIndices(BaseModel):
    """Task verification index mapping entry"""
    task_id: str = Field(description="Task ID")
    verification_indices: List[int] = Field(
        description="List of subtask indices in this plan that are dedicated to verification (using original subtask numbers from the plan)",
        default_factory=list
    )


class DedicatedVerificationTaskCheckResult(BaseModel):
    """Dedicated verification task check result (batch processing of multiple plans)"""
    task_verification_map: List[TaskVerificationIndices] = Field(
        description="List of plans that contain dedicated verification subtasks",
        default_factory=list
    )
    reasons: List[TaskExplanation] = Field(
        description="For each task_id with verification tasks, explains which subtasks are verification tasks and what they verify",
        default_factory=list
    )


class ExcessiveEffortCheckResult(BaseModel):
    """Excessive effort/invocation check result"""
    excessive_effort_indices: List[int] = Field(
        description="List of subtask indices (0-based) that demand excessive reasoning, excessive tool invocations, or excessive output",
        default_factory=list
    )
    reasons: List[IndexReason] = Field(
        description="For each subtask index judged as excessive, explains why it is considered excessive",
        default_factory=list
    )


class SubtaskFlagWithReason(BaseModel):
    """Flagged subtask with its reason"""
    subtask_index: int = Field(description="Subtask number (using the original subtask number from the plan)")
    reason: str = Field(description="Reason for flagging; must point out the discrepancy between the original task wording and the subtask description")


class AssumptionInjectionCheckResult(BaseModel):
    """Prior assumption injection check result (single task-plan pair)

    Prior assumption injection: the Planner writes into a subtask information not explicitly
    given in the user's task as a premise, such as expanding the query scope, redefining the
    meaning of the user's wording, or replacing specific references with broad category searches.
    """
    flagged_subtasks: List[SubtaskFlagWithReason] = Field(
        description="List of subtasks into which the Planner injected prior assumptions; empty list if none",
        default_factory=list
    )


class VagueConcretizationCheckResult(BaseModel):
    """Vague expression over-concretization check result (single task-plan pair)

    Vague expression over-concretization: the user's task contains vague or open-ended
    expressions, and the Planner prematurely resolves the ambiguity by locking it into a
    specific interpretation that is not the only reasonable one.
    """
    flagged_subtasks: List[SubtaskFlagWithReason] = Field(
        description="List of subtasks where the Planner over-concretized the user's vague expressions; empty list if none",
        default_factory=list
    )


class PrematureCommitmentCheckResult(BaseModel):
    """Premature path commitment check result (single task-plan pair)

    Premature path commitment: without sufficient evidence, the Planner skips the step of
    'first confirming semantic anchors' and directly writes an unverified interpretation as
    fact into execution steps, instead of treating the assumption as something to be verified.
    """
    flagged_subtasks: List[SubtaskFlagWithReason] = Field(
        description="List of subtasks that reflect the Planner's premature commitment to an interpretation path; empty list if none",
        default_factory=list
    )


class MetricSelectionResult(BaseModel):
    """Metric selection result"""
    selected_metrics: List[PlanMetricType] = Field(
        description="List of selected metrics (1-2), must be chosen from the PlanMetricType enum; select the metrics most directly and centrally related to the fuzzing target"
    )
    reason: str = Field(
        description="Reasoning for selecting these metrics, explaining why they are most directly related to the fuzzing target"
    )


class PlanImprovementAnalysis(BaseModel):
    """Plan improvement analysis result (full-metrics path; LLM selects core metrics itself)"""
    has_improvement: bool = Field(
        description="Whether the mutated Agent description brings improvement, i.e. whether the relevant metrics changed toward the target direction by more than 5%"
    )
    selected_metrics: List[str] = Field(
        description="List of core metric names chosen for the judgment (typically 1-2 most relevant metrics)",
        default_factory=list
    )
    reason: str = Field(
        description="Detailed reasoning for the judgment, combining specific change values of the selected metrics and the target direction"
    )


class PlanImprovementAnalysisSelected(BaseModel):
    """Plan improvement analysis result (pre-selected metrics path; metrics are already chosen, LLM only analyzes them)"""
    has_improvement: bool = Field(
        description="Whether the mutated Agent description brings improvement, i.e. whether the provided core metrics changed toward the target direction by more than 5%"
    )
    reason: str = Field(
        description="Detailed reasoning for the judgment, combining specific change values of the provided core metrics and the target direction"
    )
