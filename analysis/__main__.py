"""
CLI entry point.

Usage:
    python -m study --task <task_name> [--platform <platform>] [--limit <N>]
"""
import logging
from argparse import ArgumentParser

# Import task modules so their @task decorators register before we dispatch.
import analysis.tasks  # noqa: F401

from analysis.analyzer import Analyzer, _TASK_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = ArgumentParser(description="Structured LLM study of agent data")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(_TASK_REGISTRY) or None,
        help="Task to run",
    )
    parser.add_argument("--platform", default=None, help="Limit to one platform")
    parser.add_argument("--limit", type=int, default=None, help="Max agents to process")
    args = parser.parse_args()

    analyzer = Analyzer()
    try:
        analyzer.run(args.task, args.platform, args.limit)
    finally:
        analyzer.close()
