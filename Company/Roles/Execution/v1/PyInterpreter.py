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

mcp_timeout = 1200

NAME = "PythonExecutor"
DESCRIPTION = (
    "An agent that can execute Python code snippets in an isolated container environment, "
    "similar to running cells in a Jupyter Notebook. "
    "Its execution is strictly limited to the Python code and data explicitly provided to it. "
    "Useful for performing math calculations or demonstrating small code examples."
)


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

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
                "mcp/mcp-code-interpreter:latest",
            ],
            read_timeout_seconds=mcp_timeout
        )

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message="""
You are the PythonExecutor agent.

Your sole responsibility is to execute Python code snippets and return their outputs.
Behave like a Jupyter Notebook runtime:
- Execute the provided code exactly as written.
- Capture and return stdout, stderr, and results.
- Do not perform reasoning, web access, or file operations beyond code execution.
- Keep responses concise, showing only relevant outputs or error messages.

If the code is incomplete or invalid, report the error clearly without guessing.
""",
        model_client=model_client,
        workbench=McpWorkbench(server_params=server_params),
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=20,
    )

    return agent

