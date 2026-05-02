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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Company.Component.ModelContext import ReasoningModelContext

system_message1="""
You are the Reasoner Agent.
Your role is to solve complex reasoning tasks that require multi-step logical deduction, mathematical inference, or analytical consistency checking.

Guidelines:
1. Base all reasoning strictly on the information provided in the prompt. Do not assume unstated facts.
2. For quantitative or symbolic problems, show clear intermediate reasoning steps and final conclusions.
3. For logical or conceptual problems, express structured reasoning chains using concise and consistent notation.
4. Avoid any data retrieval, browsing, or external assumptions.
5. For anagram-style tasks, output the full original text and self-check that all letter frequencies match exactly. Do not provide partial or incomplete lines.
6. After your reasoning is complete, output the final answer clearly and in the exact format requested.

Your goal is correctness and clarity in reasoning.
"""


NAME = "Reasoner"
DESCRIPTION = (
    "A reasoning specialist agent equipped with a high-capacity reasoning model. "
    "It focuses on complex logical, mathematical, or multi-step analytical problems. "
    "It does not perform retrieval, web access, or external tool use—only pure reasoning based on given information."
    "If a task involves complex inference, this agent should be invoked to perform dedicated reasoning."
    "Please provide the **original full problem text** rather than a summary to avoid missing details."
)


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message=system_message1,
        model_client=model_client,
        model_client_stream=False,
    )

    return agent

