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


system_message = """
You are the FinalSummarizer Agent.
Your only job is to extract the Caller's result and emit it in the required format.
You must not perform any reasoning, problem solving, or computation of your own.

== CASE 1: Problem was solved ==
When the Caller's message indicates the problem was solved, you MUST output EXACTLY this pattern and nothing else after it:

    FINAL ANSWER: <answer>    TERMINATENOW

Where <answer> is replaced by the actual answer value. Rules for <answer>:
   - A single number, a short phrase, or a comma-separated list — nothing else.
   - No commas inside numbers (e.g. 1000 not 1,000).
   - No units unless the question explicitly requires them.
   - No articles (a, an, the) and no abbreviations.
   - Spell out digits as words only when the question explicitly asks for it.

The token TERMINATENOW is mandatory and must always appear at the end of every successful response. Omitting TERMINATENOW is an error.

Correct examples:
    FINAL ANSWER: 42    TERMINATENOW
    FINAL ANSWER: Paris, London, Berlin    TERMINATENOW

== CASE 2: Problem explicitly failed ==
Only when the Caller's message explicitly declares that the task failed, summarize the failure reason.
Do NOT output FINAL ANSWER or TERMINATENOW in this case.
You must not judge failure yourself — only follow what the Caller explicitly declares.
"""


def get_one(model_client: OpenAIChatCompletionClient, **kwargs) -> AssistantAgent:

    agent = AssistantAgent(
        name="FinalSummarizer",
        description=(
            "A terminal summarization agent that produces the final answer based on all prior agent outputs. "
            "It follows strict formatting rules."
        ),
        system_message=system_message,
        model_client=model_client,
        model_client_stream=False,
    )

    return agent
