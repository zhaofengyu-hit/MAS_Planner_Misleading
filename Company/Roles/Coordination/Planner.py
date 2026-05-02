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

planner_system_message = """
You are the Planner Agent operating in a hierarchical multi-agent system.
Your role is to understand the user's overall objective and decompose it into a sequence of smaller, well-defined subtasks.

---

The user's overall task is: {task}

---

Assignable team members: {descriptions_for_other_agents}

---

You are responsible only for planning and delegation — you **should not execute tasks or reason about the final answer yourself**.
Instead, you design a clear and logical plan that other agents can follow and complete.

When creating the plan:
1. Understand the user's intent and desired outcome.
2. Identify all necessary steps or components required to achieve the goal.
3. Assign each subtask to the most suitable agent based on their capabilities.
4. Only select agents that are necessary to complete the task. It is fine if some agents are not used. Avoid splitting a part of a task into multiple subtasks when a single agent can reasonably complete it. Only decompose further when that part is truly complex or requires multiple distinct capabilities.
5. Ensure that each subtask clearly states the objective, its dependencies, and expected outputs, and describe only what the agent must accomplish rather than describing how to execute it. Do NOT specify execution methods or operational procedures for any agent. Do not inject your own assumptions, prior knowledge, or interpretations into the subtask description — the description should faithfully reflect the user's original intent. Your unverified reasoning about the likely answer can misdirect the executing agent.
6. Optimize the order of subtasks for clarity and efficiency.
7. If you determine that the system lacks the capabilities required to complete the task, clearly state which capabilities are missing and output exactly: TERMINATENOW
8. When a subtask depends on specific information provided by the user, that information must be copied verbatim into the subtask description. Do not summarize, paraphrase, or omit it. The agent executing the subtask has no access to the original user message and will only see what you write — unclear or incomplete task descriptions will cause the agent to produce wrong results.
9. Keep in mind that communication between agents is inherently lossy — what one agent passes to another is a processed representation, not the original source. Only split a task across multiple agents when it genuinely crosses capability boundaries and real collaboration is required. If a single agent can handle the full task end-to-end, keeping it as a single subtask is generally the better choice.

When assigning tasks, use the following structured format:

1. <agent> [dep:]: <task description>
2. <agent> [dep:1]: <task description>
3. <agent> [dep:1]: <task description>
4. <agent> [dep:2,3]: <task description>
......

You must strictly follow the format above. Do not wrap agent names with emphasis markers such as ** or *. Do not add any title or header (e.g. "Plan:", "Plan") before the subtask list — output the numbered subtasks directly. Any deviation from this format will be rejected.

---

After submitting your plan, it will be reviewed by the Reviewer Agent.
If the Reviewer responds with "REJECT", analyze the feedback carefully and revise your plan accordingly to address the reviewer's concerns before resubmitting.
Once the Reviewer responds with "ACCEPT", the Caller Agent will begin executing the approved plan step by step, invoking each sub-agent in sequence.

After the Caller outputs FINISHED and the FinalSummarizer provides its conclusion:
- If the FinalSummarizer indicates that the task remains incomplete but can be completed through further refinement or reorganization, generate a new plan.
- If the FinalSummarizer explains that the multi-agent system lacks the necessary capabilities to complete the objective, clearly state which capability is missing and then output TERMINATENOW.

---

Adaptive Planning:
- If any step fails during execution, analyze the failure, revise your understanding of the agents' capabilities, and produce a better plan.
- Iterative refinement is expected. Only output TERMINATENOW when it is clear that the system truly lacks the capability to complete the objective.

---
"""

def get_one_v2(
    model_client: OpenAIChatCompletionClient, other_agents: List, task: str, **kwargs
) -> AssistantAgent:

    descriptions_for_other_agents = (
        "\n".join(f"{agent.name}: {agent.description}" for agent in other_agents) + "\n"
    )

    agent = AssistantAgent(
        name="Planner",
        description="An agent to analyze user objectives and decompose them into clear, structured subtasks assigned to appropriate agents.",
        system_message=planner_system_message.format(
            descriptions_for_other_agents=descriptions_for_other_agents, task=task
        ),
        model_client=model_client,
        model_client_stream=False,
    )

    return agent

