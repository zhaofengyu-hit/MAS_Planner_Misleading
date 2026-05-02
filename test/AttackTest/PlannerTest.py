from autogen_ext.models.openai import OpenAIChatCompletionClient

import json
import os
import re

from pathlib import Path
from autogen_core.models import ModelFamily
import sys, os
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.task_filter import filter_task


def get_gpt_models():
    model_client = OpenAIChatCompletionClient(
        model="gpt-5",
        api_key="<gpt_api_key>",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_5,
            "structured_output": True,
        },
        # parallel_tool_calls=False,
    )
    return model_client, model_client


def get_gpt_mini_models():
    model_client = OpenAIChatCompletionClient(
        model="gpt-5-mini",
        api_key="<gpt_api_key>",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_5,
            "structured_output": True,
        },
        # parallel_tool_calls=False,
    )
    return model_client, model_client


def get_deepseek_models():
    chat_model = OpenAIChatCompletionClient(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="<deepseek_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.R1,
            "structured_output": False,
        },
        parallel_tool_calls=False,
    )
    reasoner_model = OpenAIChatCompletionClient(
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        api_key="<deepseek_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.R1,
            "structured_output": False,
        },
        parallel_tool_calls=False,
    )
    return chat_model, reasoner_model


def get_deepseek_models_ali():
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model = OpenAIChatCompletionClient(
        model="deepseek-v3.2",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.R1,
            "structured_output": False,
        },
        parallel_tool_calls=False,
    )
    reasoner_model = OpenAIChatCompletionClient(
        model="deepseek-v3.2",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.R1,
            "structured_output": False,
        },
        parallel_tool_calls=False,
        extra_body={"enable_thinking": True},
    )
    return chat_model, reasoner_model


def get_qwen_models():
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model = OpenAIChatCompletionClient(
        model="qwen3-max",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        parallel_tool_calls=False,
    )
    reasoner_model = OpenAIChatCompletionClient(
        model="qwen3-max",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        parallel_tool_calls=False,
        extra_body={"enable_thinking": True},
    )
    return chat_model, reasoner_model


def get_kimi_models():
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model = OpenAIChatCompletionClient(
        model="kimi-k2.5",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        parallel_tool_calls=False,
    )
    reasoner_model = OpenAIChatCompletionClient(
        model="kimi-k2.5",
        base_url=base_url,
        api_key="<ali_bailian_api_key>",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        parallel_tool_calls=False,
        extra_body={"enable_thinking": True},
    )
    return chat_model, reasoner_model


PLANNER_REGISTRY = {
    "gpt": get_gpt_models,
    "gpt_mini": get_gpt_mini_models,
    "deepseek": get_deepseek_models,
    "deepseek_ali": get_deepseek_models_ali,
    "kimi": get_kimi_models,
    "qwen": get_qwen_models,
}


async def get_plans(
    provided_desc: str,
    data_dir: str,
    subset: str = "2023_level1",
    split: str = "validation",
    use_defense: bool = False,
    cache_dir: str = None,
    planner: str = "deepseek_ali",
    worker_descs: dict[str, str] | None = None,
) -> dict:
    """
    Retrieve plans for all tasks in the dataset.

    Args:
        provided_desc: Description provided to the target agent.
        data_dir: Dataset directory.
        subset: Dataset subset (default: "2023_level1").
        split: Dataset split (default: "validation").
        use_defense: Whether to normalize provided_desc via DescriptionNormalizer.
        cache_dir: Optional checkpoint directory (e.g. tmp/plans/<seed_id>).
                   When provided, progress is saved to a single JSON file after each
                   plan and restored on restart. Without it, runs as a one-shot call.
        worker_descs: Optional dict of {agent_name: description} to override the
                      default descriptions of non-DB worker agents (e.g. WebSurfer,
                      DocumentAnalyzer, etc.). PostgresManager is always set from
                      provided_desc and is ignored if present in this dict.

    Returns:
        dict with keys:
            - members: {agent_name: agent_description}
            - plans:   {task_id: plan_text}
            - tasks:   {task_id: task_description}
    """
    from Company.Teams import ExploitableTeam

    if use_defense:
        from Company.Defense.normalizer import DescriptionNormalizer
        normalizer = DescriptionNormalizer()
        agent_descs = {"PostgresManager": normalizer.normalize(provided_desc)}
        print("Using defense with normalized description:")
        print(agent_descs["PostgresManager"])
    else:
        agent_descs = {"PostgresManager": provided_desc}

    if worker_descs:
        for k, v in worker_descs.items():
            if k != "PostgresManager":
                agent_descs[k] = v

    chat_model, reasoner_model = get_deepseek_models_ali()
    planner_model, _ = PLANNER_REGISTRY.get(planner, get_deepseek_models_ali)()
    print(f"Using planner model: {planner}")

    dataset = load_dataset(data_dir, subset, split=split)

    members = None
    plans = {}
    tasks = {}

    # Load checkpoint if cache_dir is provided
    cache_path = Path(cache_dir) if cache_dir else None
    checkpoint_file = cache_path / "checkpoint.json" if cache_path else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        if checkpoint_file.exists():
            with open(checkpoint_file, encoding="utf-8") as f:
                ckpt = json.load(f)
            members = ckpt.get("members")
            plans = dict(ckpt.get("plans", {}))
            tasks = dict(ckpt.get("tasks", {}))
            print(f"[checkpoint] restored {len(plans)} plans from {checkpoint_file}")

    def _save_checkpoint():
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {"members": members, "plans": plans, "tasks": tasks},
                f,
                ensure_ascii=False,
                indent=2,
            )

    for example in dataset:
        question = example["Question"]
        qid = example["task_id"]
        file_path = (
            os.path.join(data_dir, example["file_path"])
            if example["file_path"]
            else None
        )
        if file_path:
            question += f" (The related file is at {file_path}.)"

        if not filter_task(qid):
            continue

        if qid in plans:
            print(f"[cache hit] {qid}")
            continue

        team, tool_agents_dict = await ExploitableTeam.get_plan_only_team_with_desc(
            chat_model,
            reasoner_model,
            question,
            planner_model=planner_model,
            agent_descs=agent_descs,
        )

        if members is None:
            members = tool_agents_dict.copy()

        plan = None
        for attempt in range(3):
            if attempt > 0:
                print(f"[retry {attempt}] {qid}")
                team, _ = await ExploitableTeam.get_plan_only_team_with_desc(
                    chat_model,
                    reasoner_model,
                    question,
                    planner_model=planner_model,
                    agent_descs=agent_descs,
                )

            result = await team.run(task=question)
            messages = result.messages

            # Third-to-last message is the planner's plan
            # (last = terminator, second-to-last = reviewer)
            if len(messages) >= 3:
                plan = messages[-3].content
            else:
                plan = None

            if plan and re.search(r"^\d+\.\s+\S", plan, re.MULTILINE):
                break
            print(f"[invalid plan, attempt {attempt + 1}] {qid}: {plan}")

        plans[qid] = plan
        tasks[qid] = question

        if cache_path:
            _save_checkpoint()

    return {"members": members, "plans": plans, "tasks": tasks}
