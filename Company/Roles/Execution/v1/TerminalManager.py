from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams, StdioServerParams

from autogen_agentchat.ui import Console

import asyncio

import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
    ModelFamily,
)
from autogen_core.tools import ToolResult, Workbench
from autogen_agentchat.messages import TextMessage

from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

import os, sys

shared_folder = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    + "/Docker/shared_folder/"
)

mcp_timeout = 1200


NAME = "TerminalManager"
DESCRIPTION = (
    "An agent that executes shell commands inside an isolated Docker container environment, completely separated from the host system. "
    "It performs command-line tasks such as file operations, simple scripting, and other actions suitable for a terminal. "
    "It focuses solely on completing the tasks assigned to it through direct command execution. "
)


def get_one_v2(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    if "url" in kwargs:
        server_params = SseServerParams(
            url=kwargs["url"], timeout=mcp_timeout, sse_read_timeout=mcp_timeout
        )
    else:
        server_params = StdioServerParams(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                # "-v",
                # f"{shared_folder}:/home/shared_folder/",
                "astral/uv:python3.14-bookworm",
                "uvx",
                "terminal_controller",
            ],
            read_timeout_seconds=mcp_timeout
        )

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message="""
You are the TerminalManager agent.

Your responsibility is to execute terminal commands inside an isolated environment.  
You can interpret general user intents and execute corresponding shell commands to fulfill them.

Capabilities:
1. Run shell commands to perform file operations, retrieve resources, manage directories, or execute simple scripts.
2. Infer reasonable commands when the user gives high-level or imprecise instructions.

Guidelines:
- If a command fails, report the error and the reason when available.
- Do not provide explanations, reasoning, commentary, or advice unrelated to command execution.
- Avoid any actions that could harm the system or access data outside the permitted environment.

Act as a reliable, minimal terminal operator that translates assigned tasks into safe, concrete command executions.
""",
        model_client=model_client,
        workbench=McpWorkbench(server_params=server_params),
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=20,
    )

    return agent

