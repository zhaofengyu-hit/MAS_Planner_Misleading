from pathlib import Path
import json
import re
from enum import Enum


class LLMCallPassedTool:
    def __init__(self, tool: dict):
        self.func_name: str = ""
        self.func_desc: str = ""

        assert tool["type"] == "function"
        self.func_name = tool["function"]["name"]
        self.func_desc = tool["function"]["description"]


class ToolCall:
    def __init__(self, tool_call: dict):
        self.tool_call_id: str = ""
        self.func_name: str = ""
        self.func_args: str = ""

        self.tool_call_id = tool_call["id"]
        self.func_name = tool_call["function"]["name"]
        self.func_args = tool_call["function"]["arguments"]


class LLMCallPassedMessage:
    def __init__(self, message: dict):
        self.role: str = ""
        self.content: str = ""
        self.tool_call_id: str = ""
        self.tool_calls: list[ToolCall] = []

        self.role = message["role"]
        self.content = message["content"] if message["content"] is not None else ""
        if "tool_call_id" in message:
            self.tool_call_id = message["tool_call_id"]
        if "tool_calls" in message:
            for item in message["tool_calls"]:
                new_tool_call = ToolCall(item)
                self.tool_calls.append(new_tool_call)


class LLMCallUsage:
    def __init__(self, usage: dict):
        self.completion_tokens: int = 0
        self.prompt_tokens: int = 0
        self.total_tokens: int = 0
        self.completion_accepted_prediction_tokens: int = 0
        self.completion_audio_tokens: int = 0
        self.completion_reasoning_tokens: int = 0
        self.completion_rejected_prediction_tokens: int = 0
        self.prompt_audio_tokens: int = 0
        self.prompt_cached_tokens: int = 0

        self.completion_tokens = int(usage["completion_tokens"] or 0)
        self.prompt_tokens = int(usage["prompt_tokens"] or 0)
        self.total_tokens = int(usage["total_tokens"] or 0)

        prompt_details = usage["prompt_tokens_details"] or {}
        self.prompt_audio_tokens = int(prompt_details.get("audio_tokens") or 0)
        self.prompt_cached_tokens = int(prompt_details.get("cached_tokens") or 0)

        completion_details = usage["completion_tokens_details"] or {}
        self.completion_accepted_prediction_tokens = int(
            completion_details.get("accepted_prediction_tokens") or 0
        )
        self.completion_audio_tokens = int(completion_details.get("audio_tokens") or 0)
        self.completion_reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
        self.completion_rejected_prediction_tokens = int(
            completion_details.get("rejected_prediction_tokens") or 0
        )


finish_reasons = set(["stop", "tool_calls"])


class LLMCallResponse:
    def __init__(self, response: dict):
        self.reason: str = ""
        self.role: str = ""
        self.content: str = ""
        self.tool_calls: list[ToolCall] = []
        self.timestamp: int = 0
        self.model: str = ""
        self.usage: LLMCallUsage = None

        self.reason = response["choices"][0]["finish_reason"]
        self.content = (
            response["choices"][0]["message"]["content"]
            if response["choices"][0]["message"]["content"] is not None
            else ""
        )
        self.role = response["choices"][0]["message"]["role"]

        if self.reason == "tool_calls":
            for item in response["choices"][0]["message"]["tool_calls"]:
                new_tool_call = ToolCall(item)
                self.tool_calls.append(new_tool_call)

        if self.reason not in finish_reasons:
            print(f"Other Response Finish Reason: {self.reason}")

        self.timestamp = int(response["created"])
        self.model = response["model"]
        self.usage = LLMCallUsage(response["usage"])

    def is_tool_call_response(self):
        return self.reason == "tool_calls"


class LLMCallEventInfo:
    def __init__(self, event: dict):
        self.agent_name: str = ""
        self.tools: list[LLMCallPassedTool] = []
        self.messages: list[LLMCallPassedMessage] = []
        self.response: LLMCallResponse = None

        for item in event["messages"]:
            new_message = LLMCallPassedMessage(item)
            self.messages.append(new_message)
        for item in event["tools"]:
            new_tool = LLMCallPassedTool(item)
            self.tools.append(new_tool)
        self.response = LLMCallResponse(event["response"])

        self._extract_agent_name()
        assert self.agent_name != ""

    def _extract_agent_name(self):
        system_message = self.messages[0]
        if system_message.content.startswith("Select an agent"):
            self.agent_name = "Selector"
            return
        pattern = r"you are the (\w+) agent"
        match = re.search(pattern, system_message.content, re.IGNORECASE)

        if match:
            self.agent_name = match.group(1)

    def get_call_token_num(self):
        return self.response.usage.total_tokens

    def is_tool_call(self):
        return self.response.is_tool_call_response()

    def is_the_agent(self, agent_name: str):
        return self.agent_name.lower() == agent_name.lower()

    def get_subtask_num(self):
        if self.is_the_agent("planner"):
            pattern = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
            return len(pattern.findall(self.response.content))
        else:
            return 0


class TraceLog:
    def __init__(self, logfile_path: Path):
        self.start_time: float = 0
        self.end_time: float = 0
        self.events: list[LLMCallEventInfo] = []

        assert logfile_path.exists()

        with open(logfile_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

            self.start_time = float(log_data[-1]["start_time"])
            self.end_time = float(log_data[-1]["end_time"])

            for item in log_data[:-1]:
                new_event = LLMCallEventInfo(item)
                self.events.append(new_event)

    def get_execution_seconds(self):
        return self.end_time - self.start_time

    def get_llm_call_num(self):
        return len(self.events)

    def get_total_token_num(self):
        total_num = 0
        for item in self.events:
            total_num += item.get_call_token_num()
        return total_num

    def get_tool_call_num(self):
        total_num = 0
        for item in self.events:
            total_num += 1 if item.is_tool_call() else 0
        return total_num

    def get_subtask_num(self):
        subtask_num = 0
        accepted_plan_num = 0
        plan_indexes: list[int] = []
        for i in range(len(self.events)):
            if self.events[i].is_the_agent("planner"):
                plan_indexes.append(i)
        for index in plan_indexes:
            reviewer_index = index + 2
            if reviewer_index >= len(self.events):
                break
            reviewer_llm_call = self.events[reviewer_index]
            if not reviewer_llm_call.is_the_agent("reviewer"):
                continue
            resp_content = reviewer_llm_call.response.content
            if resp_content.lower() != "accept":
                continue
            caller_index = reviewer_index + 2
            if caller_index >= len(self.events):
                break
            caller_llm_call = self.events[caller_index]
            if not caller_llm_call.is_the_agent("caller"):
                continue

            planner_llm_call = self.events[index]
            subtask_num += planner_llm_call.get_subtask_num()
            accepted_plan_num += 1

            # break
        result = subtask_num
        if subtask_num != 0:
            result = subtask_num / accepted_plan_num
        return result


class TaskInfo:
    def __init__(self, task: dict):
        self.id: str = ""
        self.level: int = 0
        self.question: str = ""
        self.expected_answer: str = ""
        self.agent_answer: str = ""
        self.success: bool = False

        self.id = task["id"]
        self.level = task["level"]
        self.question = task["question"]
        self.expected_answer = task["expected_answer"]
        self.agent_answer = task["agent_answer"]
        self.success = task["judgement"]["pass"]


class TestResult:
    def __init__(self, file_path: Path):
        self.tasks: dict[str, TaskInfo] = {}

        assert file_path.exists()

        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            for id, item in content["run_cache"].items():
                new_task_info = TaskInfo(item)
                self.tasks[id] = new_task_info


class MetricsCalculator:
    def __init__(self, task_logs: dict[str, TraceLog], test_info: TestResult):
        self.task_logs: dict[str, TraceLog] = {}
        self.test_info: TestResult = None

        self.task_logs = task_logs
        self.test_info = test_info

    def _success_rate(self) -> float:
        success_num = 0
        for _, item in self.test_info.tasks.items():
            if item.success:
                success_num += 1
        total_num = len(self.test_info.tasks)

        if total_num == 0:
            return 0.0

        success_rate = (success_num / total_num) * 100
        return round(success_rate, 2)

    def _avg_time_per_task(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        total_time = 0.0
        for task_id, trace_log in self.task_logs.items():
            task_time = trace_log.get_execution_seconds()
            total_time += task_time

        avg_time = total_time / len(self.task_logs)
        return round(avg_time, 2)

    def _avg_llm_calls_per_task(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        total_calls = 0
        for task_id, trace_log in self.task_logs.items():
            total_calls += trace_log.get_llm_call_num()

        avg_calls = total_calls / len(self.task_logs)
        return round(avg_calls, 2)

    def _avg_tokens_per_task(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        total_tokens = 0
        for task_id, trace_log in self.task_logs.items():
            total_tokens += trace_log.get_total_token_num()

        avg_tokens = total_tokens / len(self.task_logs)
        return round(avg_tokens, 2)

    def _avg_tool_calls_per_task(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        total_tool_calls = 0
        for task_id, trace_log in self.task_logs.items():
            total_tool_calls += trace_log.get_tool_call_num()

        avg_tool_calls = total_tool_calls / len(self.task_logs)
        return round(avg_tool_calls, 2)

    def _avg_subtasks_per_task(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        total_subtasks = 0
        for task_id, trace_log in self.task_logs.items():
            total_subtasks += trace_log.get_subtask_num()

        avg_subtasks = total_subtasks / len(self.task_logs)
        return round(avg_subtasks, 2)

    def _avg_tokens_per_llm_call(self) -> float:
        if len(self.task_logs) == 0:
            return 0.0

        avg_tokens_per_task = []
        for task_id, trace_log in self.task_logs.items():
            llm_call_num = trace_log.get_llm_call_num()
            if llm_call_num == 0:
                continue
            total_tokens = trace_log.get_total_token_num()
            avg_tokens = total_tokens / llm_call_num
            avg_tokens_per_task.append(avg_tokens)

        if len(avg_tokens_per_task) == 0:
            return 0.0

        overall_avg = sum(avg_tokens_per_task) / len(avg_tokens_per_task)
        return round(overall_avg, 2)

    def _avg_tokens_per_agent(self) -> dict[str, float]:
        agent_tokens_map: dict[str, list[int]] = {}

        for task_id, trace_log in self.task_logs.items():
            task_agent_tokens: dict[str, int] = {}

            for event in trace_log.events:
                agent_name = event.agent_name
                token_num = event.get_call_token_num()

                if agent_name not in task_agent_tokens:
                    task_agent_tokens[agent_name] = 0
                task_agent_tokens[agent_name] += token_num

            for agent_name, tokens in task_agent_tokens.items():
                if agent_name not in agent_tokens_map:
                    agent_tokens_map[agent_name] = []
                agent_tokens_map[agent_name].append(tokens)

        result: dict[str, float] = {}
        for agent_name, token_list in agent_tokens_map.items():
            avg_tokens = sum(token_list) / len(token_list)
            result[agent_name] = round(avg_tokens, 2)

        return result

    def _avg_tool_calls_per_agent(self) -> dict[str, float]:
        agent_tool_calls_map: dict[str, list[int]] = {}

        for task_id, trace_log in self.task_logs.items():
            task_agent_tool_calls: dict[str, int] = {}

            for event in trace_log.events:
                agent_name = event.agent_name
                if event.is_tool_call():
                    if agent_name not in task_agent_tool_calls:
                        task_agent_tool_calls[agent_name] = 0
                    task_agent_tool_calls[agent_name] += len(event.response.tool_calls)

            for agent_name, tool_calls in task_agent_tool_calls.items():
                if agent_name not in agent_tool_calls_map:
                    agent_tool_calls_map[agent_name] = []
                agent_tool_calls_map[agent_name].append(tool_calls)

        result: dict[str, float] = {}
        for agent_name, tool_calls_list in agent_tool_calls_map.items():
            if len(tool_calls_list) == 0:
                continue
            avg_tool_calls = sum(tool_calls_list) / len(tool_calls_list)
            result[agent_name] = round(avg_tool_calls, 2)

        return result

    def _agent_participation_rate(self) -> dict[str, float]:
        if len(self.task_logs) == 0:
            return {}

        agent_task_count: dict[str, int] = {}

        for _, trace_log in self.task_logs.items():
            agents_in_task = set()
            for event in trace_log.events:
                agents_in_task.add(event.agent_name)

            for agent_name in agents_in_task:
                if agent_name not in agent_task_count:
                    agent_task_count[agent_name] = 0
                agent_task_count[agent_name] += 1

        total_tasks = len(self.task_logs)
        result: dict[str, float] = {}
        for agent_name, count in agent_task_count.items():
            participation_rate = (count / total_tasks) * 100
            result[agent_name] = round(participation_rate, 2)

        return result


class MetricType(str, Enum):
    # Basic metrics (from MetricsCalculator methods)
    SUCCESS_RATE = "success_rate"
    AVG_TIME_PER_TASK = "avg_time_per_task"
    AVG_LLM_CALLS_PER_TASK = "avg_llm_calls_per_task"
    AVG_TOKENS_PER_TASK = "avg_tokens_per_task"
    AVG_TOOL_CALLS_PER_TASK = "avg_tool_calls_per_task"
    AVG_SUBTASKS_PER_TASK = "avg_subtasks_per_task"
    AVG_TOKENS_PER_LLM_CALL = "avg_tokens_per_llm_call"

    # Agent metrics (not individual values, but categories)
    AVG_TOKENS_PER_AGENT = "avg_tokens_per_agent"
    AVG_TOOL_CALLS_PER_AGENT = "avg_tool_calls_per_agent"
    AGENT_PARTICIPATION_RATE = "agent_participation_rate"

def get_metrics_report(
    result_dir_path: str,
    exclude_task_ids: set[str] | None = None,
) -> dict:
    excluded_agents = {"Planner", "Reviewer", "Caller", "FinalSummarizer", "Selector"}

    result_dir = Path(result_dir_path)
    test_result, task_logs = parse_result(result_dir, exclude_task_ids=exclude_task_ids)
    mc = MetricsCalculator(task_logs=task_logs, test_info=test_result)

    participation_rate = mc._agent_participation_rate()
    filtered_participation = {
        name: rate
        for name, rate in participation_rate.items()
        if name not in excluded_agents
    }

    avg_tokens_per_agent = mc._avg_tokens_per_agent()
    filtered_tokens = {
        name: tokens
        for name, tokens in avg_tokens_per_agent.items()
        if name not in excluded_agents
    }

    avg_tool_calls_per_agent = mc._avg_tool_calls_per_agent()
    filtered_tool_calls = {
        name: calls
        for name, calls in avg_tool_calls_per_agent.items()
        if name not in excluded_agents
    }

    report = {
        "basic_metrics": {
            MetricType.SUCCESS_RATE.value: {
                "description": "Task success rate (percentage)",
                "value": mc._success_rate(),
            },
            MetricType.AVG_TIME_PER_TASK.value: {
                "description": "Average execution time per task (seconds)",
                "value": mc._avg_time_per_task(),
            },
            MetricType.AVG_LLM_CALLS_PER_TASK.value: {
                "description": "Average number of LLM calls per task",
                "value": mc._avg_llm_calls_per_task(),
            },
            MetricType.AVG_TOKENS_PER_TASK.value: {
                "description": "Average number of tokens consumed per task",
                "value": mc._avg_tokens_per_task(),
            },
            MetricType.AVG_TOOL_CALLS_PER_TASK.value: {
                "description": "Average number of tool calls per task",
                "value": mc._avg_tool_calls_per_task(),
            },
            MetricType.AVG_SUBTASKS_PER_TASK.value: {
                "description": "Average number of subtasks decomposed per task",
                "value": mc._avg_subtasks_per_task(),
            },
            MetricType.AVG_TOKENS_PER_LLM_CALL.value: {
                "description": "Average number of tokens consumed per LLM call",
                "value": mc._avg_tokens_per_llm_call(),
            },
        },
        "agent_metrics": {
            MetricType.AGENT_PARTICIPATION_RATE.value: {
                "description": "Proportion of tasks each agent participates in (percentage)",
                "value": filtered_participation,
            },
            MetricType.AVG_TOKENS_PER_AGENT.value: {
                "description": "Average tokens consumed by each agent across the tasks it participates in",
                "value": filtered_tokens,
            },
            MetricType.AVG_TOOL_CALLS_PER_AGENT.value: {
                "description": "Average number of tool calls made by each agent across the tasks it participates in",
                "value": filtered_tool_calls,
            },
        },
    }

    return report


def get_total_tokens_in_directory(result_dir_path: str) -> dict:
    """Count total tokens consumed by all tasks in a directory.

    Args:
        result_dir_path: Path to the result directory.

    Returns:
        Dict with token statistics:
        - total_tokens: total tokens across all tasks
        - total_prompt_tokens: total prompt tokens across all tasks
        - total_completion_tokens: total completion tokens across all tasks
        - task_count: number of tasks
        - per_task: per-task token consumption details
    """
    result_dir = Path(result_dir_path)
    assert result_dir.exists() and result_dir.is_dir(), f"Directory not found: {result_dir}"

    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    per_task_tokens: dict[str, dict] = {}

    for json_file in result_dir.glob("*.json"):
        if json_file.name == "task_cache.json":
            continue

        task_id = json_file.stem
        if task_id.endswith("_llm_call_log"):
            task_id = task_id[:-len("_llm_call_log")]

        try:
            trace_log = TraceLog(json_file)
            task_total = 0
            task_prompt = 0
            task_completion = 0

            for event in trace_log.events:
                usage = event.response.usage
                task_total += usage.total_tokens
                task_prompt += usage.prompt_tokens
                task_completion += usage.completion_tokens

            total_tokens += task_total
            total_prompt_tokens += task_prompt
            total_completion_tokens += task_completion

            per_task_tokens[task_id] = {
                "total_tokens": task_total,
                "prompt_tokens": task_prompt,
                "completion_tokens": task_completion,
            }
        except Exception as e:
            print(f"Failed to parse file {json_file}: {e}")
            continue

    return {
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "task_count": len(per_task_tokens),
        "per_task": per_task_tokens,
    }


def calculate_cost(token_report: dict, input_price: float = 1.25, output_price: float = 10.0) -> dict:
    """Calculate cost in USD from a token statistics report.

    Args:
        token_report: Report returned by get_total_tokens_in_directory.
        input_price: Price per 1M input tokens (USD); default $1.25/1M (GPT-5).
        output_price: Price per 1M output tokens (USD); default $10.00/1M (GPT-5).

    Returns:
        Dict with cost information:
        - input_cost: cost of input tokens (USD)
        - output_cost: cost of output tokens (USD)
        - total_cost: total cost (USD)
        - prompt_tokens: number of input tokens
        - completion_tokens: number of output tokens
    """
    prompt_tokens = token_report["total_prompt_tokens"]
    completion_tokens = token_report["total_completion_tokens"]

    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": token_report["total_tokens"],
    }


def parse_result(
    result_dir: Path,
    exclude_task_ids: set[str] | None = None,
) -> tuple[TestResult, dict[str, TraceLog]]:
    assert result_dir.exists() and result_dir.is_dir(), f"dir not exists: {result_dir}"

    task_cache_path = result_dir / "task_cache.json"
    assert task_cache_path.exists(), f"task_cache.json not exists: {task_cache_path}"
    test_result = TestResult(task_cache_path)

    if exclude_task_ids:
        test_result.tasks = {
            tid: t for tid, t in test_result.tasks.items() if tid not in exclude_task_ids
        }

    task_logs: dict[str, TraceLog] = {}

    for json_file in result_dir.glob("*.json"):
        if json_file.name == "task_cache.json":
            continue

        task_id = json_file.stem
        if task_id.endswith("_llm_call_log"):
            task_id = task_id[:-len("_llm_call_log")]

        if exclude_task_ids and task_id in exclude_task_ids:
            continue

        trace_log = TraceLog(json_file)
        task_logs[task_id] = trace_log

    return test_result, task_logs


if __name__ == "__main__":
    result_dir = (
        Path(__file__).parent.parent.parent.parent
        / "tmp"
        / "logs"
        / "1773898895_96c180ce_with_defense"
        # / "1773349146_8f78da57"
        # / "normal_desc_results_deepseek_ali_5"
    )
    report = get_metrics_report(str(result_dir))
    print(report)
    # print(json.dumps(report, indent=2, ensure_ascii=False))
    result = get_total_tokens_in_directory(result_dir)
    result = calculate_cost(result, input_price=2, output_price=3)
    print(result)
