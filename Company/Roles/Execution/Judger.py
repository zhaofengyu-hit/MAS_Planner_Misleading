from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams, StdioServerParams

from autogen_agentchat.ui import Console

import asyncio

import json
from dataclasses import dataclass
from typing import List, Literal

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
from pydantic import BaseModel

class AgentResponse(BaseModel):
    result: Literal["pass", "not pass"]
    reason: str

def get_one(model_client: OpenAIChatCompletionClient, **kwargs) -> AssistantAgent:

    agent = AssistantAgent(
        name="Judger",
        description="An evaluation agent that judges whether an answer correctly matches the expected answer and explains why.",
        system_message="""
You are the Judger Agent responsible for evaluating QA results.
Given a question, an expected answer, and an actual answer, your job is to decide if the result is PASS or NOT PASS, and explain briefly why.

Guidelines:
1. Read the question, expected answer, and actual answer carefully.
2. Determine if the actual answer conveys the same meaning or value as the expected one.
   - Minor wording or formatting differences are acceptable.
   - For numeric answers, allow small rounding or unit-conversion differences.
3. If the answer is semantically equivalent to the correct result, including cases where different units are used but are convertible to the same value, consider the answer correct.
3. Output ONLY a raw JSON object with the following format:
   {
     "pass": true/false,
     "reason": "concise explanation"
   }
4. Keep the reason short and factual.
5. CRITICAL: Your entire response must be ONLY the JSON object itself.
   - Do NOT wrap it in markdown code blocks (no ```json or ``` markers).
   - Do NOT add any text before or after the JSON.
   - Your response must start with '{' and end with '}'.
   - Any deviation from this format will cause a system failure.
6. When matching the final answer, note that **FINAL ANSWER** and **TERMINATENOW** are normal markers used by the system. The former indicates where the answer appears, and the latter signals the end of output. Ignore both during evaluation.
""",
        model_client=model_client,
        model_client_stream=False,
    )

    return agent
