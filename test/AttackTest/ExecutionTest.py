import asyncio
import os, shutil
import sys
import json
import argparse
from datasets import load_dataset
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.messages import ModelClientStreamingChunkEvent
from autogen_agentchat.base import TaskResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.task_filter import filter_task
from Company.Teams import ExploitableTeam
from Company.Roles.Execution import Judger
from Company.logger import LLMEventLogger
from Company.Defense.normalizer import DescriptionNormalizer

event_logger = LLMEventLogger()

workspace_dir = Path(__file__).parent.parent.parent.joinpath("tmp/").absolute()


def cleanup_workspace():
    for subdir in ["downloads", "browser_wp", "keyframes"]:
        target = workspace_dir / subdir
        if target.exists():
            for item in target.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()


skip_list = [
    "72e110e7-464c-453c-a309-90a95aed6538",
    "65afbc8a-89ca-4ad5-8d62-355bb401f61d",
]


def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_file: str, cache: dict) -> None:
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)


async def solve_one_question(
    run_task: callable, question: str, file_path: str | None
) -> dict:
    if file_path:
        question += f" (The related file is at {file_path}.)"
    messages = await run_task(question)
    return messages[-1].content


def evaluate_gaia_dataset(
    run_task: callable,
    judge: callable,
    data_dir: str,
    cache: dict,
    cache_file_path: str,
    log_dir: str,
    subset: str = "2023_level1",
    split: str = "validation",
) -> None:

    run_cache = cache.setdefault("run_cache", {})
    dataset = load_dataset(data_dir, subset, split=split)
    for example in dataset:
        question = example["Question"]
        qid = example["task_id"]

        if not filter_task(qid):
            continue

        if qid in skip_list:
            continue

        if qid in run_cache:
            continue

        file_path = (
            os.path.join(data_dir, example["file_path"])
            if example["file_path"]
            else None
        )
        expected_answer = example["Final answer"]
        cleanup_workspace()

        log_file_path = os.path.join(log_dir, f"{qid}_llm_call_log.json")

        event_logger.start()
        future = solve_one_question(run_task, question, file_path)
        actual_answer = asyncio.run(future)
        event_logger.stop()
        event_logger.save(log_file_path)
        event_logger.reset_tracker()

        future = judge(question, expected_answer, actual_answer)
        judgement = asyncio.run(future)

        record = {
            "id": qid,
            "level": example["Level"],
            "question": question,
            "file_path": file_path,
            "expected_answer": expected_answer,
            "agent_answer": actual_answer,
            "judgement": judgement,
        }

        run_cache[qid] = record
        save_cache(cache_file_path, cache)


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


def get_judger_model():
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
    return chat_model


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


def main(
    provided_desc: str,
    content_dir: str,
    use_defense: bool = False,
) -> None:
    chat_model, reasoner_model = get_deepseek_models_ali()
    planner_model = None

    if use_defense:
        normalizer = DescriptionNormalizer()
        agent_descs = {"PostgresManager": normalizer.normalize(provided_desc)}
        print("Using defense with normalized description:")
        print(agent_descs["PostgresManager"])
    else:
        agent_descs = {"PostgresManager": provided_desc}

    async def run_team(task: str) -> list:
        team = await ExploitableTeam.get_selector_team_with_dbmanager_desc(
            chat_model,
            reasoner_model,
            task,
            planner_model=planner_model,
            agent_descs=agent_descs,
        )
        stream = team.run_stream(task=task)
        messages = []
        async for message in stream:
            if isinstance(message, (ModelClientStreamingChunkEvent)):
                continue
            if isinstance(message, TaskResult):
                messages = message.messages
                continue
            print(message.source, " : ", message.content)
            print()
        return messages

    async def judge(task: str, expected_answer: str, actual_answer: str) -> dict:
        judger = Judger.get_one(model_client=get_judger_model())
        result = await judger.run(
            task=f"Question: {task}\nExpected Answer: {expected_answer}\nActual Answer: {actual_answer}\n"
        )
        try:
            result = json.loads(result.messages[-1].content)
        except Exception as e:
            print(f"Error parsing judgement result: {e}")
            print(f"Raw judgement result: {result.messages[-1].content}")
            raise e
        return result

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Benchmark", "GAIA"
    )
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tmp",
        "logs",
        content_dir,
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(log_dir, "task_cache.json")
    cache = load_cache(cache_file)

    for subset in ["2023_level1"]:
        evaluate_gaia_dataset(
            run_team,
            judge,
            data_dir,
            cache,
            cache_file,
            log_dir,
            subset=subset,
            split="validation",
        )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provided_desc",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--content_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--use_defense",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.provided_desc,
        args.content_dir,
        use_defense=args.use_defense,
    )
