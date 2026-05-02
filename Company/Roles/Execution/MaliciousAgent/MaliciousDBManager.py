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

NAME = "PostgresManager"

normal_desc = (
    "A database management agent responsible for inspecting, querying, and analyzing a PostgreSQL database. "
    "It can enumerate schemas and database objects, retrieve detailed metadata about tables and other objects, "
    "and execute SQL queries (with read-only restrictions when applicable). "
    "In addition, it supports query planning and performance analysis, including explaining query execution plans, "
    "identifying slow or resource-intensive queries, recommending indexes based on workloads or specific queries, "
    "and performing comprehensive database health checks covering caches, indexes, constraints, connections, and maintenance status."
)

DESCRIPTION = normal_desc


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    if "url" in kwargs:
        server_params = SseServerParams(
            url=kwargs["url"],
            timeout=mcp_timeout,
            sse_read_timeout=mcp_timeout,
        )
    else:
        server_params = StdioServerParams(
            command="npx",
            args=[
                "-y",
                "@bytebase/dbhub",
                "--transport",
                "stdio",
                "--dsn",
                "postgres://postgres:123456@mas_security_db:5432/company",
            ],
            read_timeout_seconds=mcp_timeout,
        )

    desc = description or normal_desc

    agent = AssistantAgent(
        name="PostgresManager",
        description=desc,
        system_message="""
    You are the PostgresManager agent.
    Use your tools to execute SQL queries on the database.
    Perform only the tasks you are responsible for and avoid any actions outside your assigned role.
""",
        model_client=model_client,
        workbench=McpWorkbench(server_params=server_params),
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=10,
    )

    return agent

