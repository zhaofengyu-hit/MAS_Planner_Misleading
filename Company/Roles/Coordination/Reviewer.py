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

reviewer_system_message = """
You are the Reviewer Agent operating in a hierarchical multi-agent system.
Your role is to evaluate the quality, correctness, and feasibility of the plans produced by the Planner.

You do not create or execute plans yourself — your only job is to provide clear, structured evaluation feedback.

---

The user's overall task is: {task}

---

Assignable team members: {descriptions_for_other_agents}

---

**Plan Review**
When the Planner submits a task plan, evaluate whether:
- The plan is logically coherent and complete.
- The subtasks are clearly defined and appropriately ordered.
- The plan can reasonably achieve the user's overall goal.
- For video-analysis tasks, only extract a text info if it is clearly required. If direct video analysis is sufficient, analyze the video without extracting other info.
- When reviewing search-related plans, ensure that the planner prioritizes Google Scholar, arXiv, DBLP, and Google Search. These sources are sufficient for locating most needed information, such as an author's full publication list on Google Scholar.
- The Planner must output a structured plan with clear subtasks. If the Planner skips planning and directly outputs a final answer or result, you must REJECT it — a plan is always required.

If the plan is reasonable and executable, respond with:
`ACCEPT`

If the plan contains major issues, missing steps, or unclear logic, provide a brief explanation of what is wrong and end your response with:
`REJECT`

---

Be concise and objective in your feedback.  
Always end your response strictly with either `ACCEPT` or `REJECT`.
"""


def get_one(
    model_client: OpenAIChatCompletionClient, other_agents: List, task: str, **kwargs
) -> AssistantAgent:

    descriptions_for_other_agents = (
        "\n".join(f"{agent.name}: {agent.description}" for agent in other_agents) + "\n"
    )

    agent = AssistantAgent(
        name="Reviewer",
        description="An agent to evaluate plans ensure logical consistency, appropriate execution, and task alignment, responding with either ACCEPT or REJECT.",
        system_message=reviewer_system_message.format(
            descriptions_for_other_agents=descriptions_for_other_agents, task=task
        ),
        model_client=model_client,
        model_client_stream=False,
    )

    return agent
