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
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.messages import TextMessage

from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Company.Component.AutoResetAgentTool import AutoResetAgentTool

caller_system_message1 = """
You are a Caller Agent operating in a hierarchical multi-agent system.
Your role is to coordinate the execution of subtasks defined by the Planner and to manage task delegation among the available agents.

Your assignable team members are: {descriptions_for_other_agents}

You are responsible for:
1. Reading and understanding the approved task plan from the Planner.
2. Selecting the next subtask to execute based on task order and dependencies.
3. Assigning the subtask to the most suitable agent from selected team members.
4. When assigning a task, you must clearly specify:
   - Which agent should perform the task.
   - What exactly the agent should do.
   - Any additional context, such as required information, previous task results, or special instructions.
5. After assigning the task, end your output. The system will automatically trigger the Reviewer Agent to evaluate the call.
6. If the Reviewer responds with "REJECT", revise your task assignment according to the feedback and resubmit.
   If the Reviewer responds with "ACCEPT", the assigned agent will proceed to execute the task automatically.
7. Continue this process until all subtasks in the plan have been reviewed and executed. Once all tasks are completed successfully, output:TERMINATE

You do not execute any tasks yourself.
Your responsibility is to manage the flow of execution and ensure each subtask is properly delegated and reviewed before moving on.
"""


caller_system_message2 = """
You are the Caller Agent in a hierarchical multi-agent system.
You do not converse directly with other agents. Instead, you invoke them as tools through function calls.

## ABSOLUTE RULE — read this before anything else
Each agent you invoke is completely isolated. It has NO access to conversation history, NO memory of what previous agents did, and can ONLY see the exact text you write into the tool input field — nothing more.
This means: if a previous agent produced output that the current subtask depends on, you MUST copy that output in full, word for word, into the tool input. There are NO exceptions.
The following are strictly FORBIDDEN in any input:
- "[Full changelog text as above]"
- "[as extracted in step 1]"
- "[see previous output]"
- "[full content as above]"
- Any other placeholder, reference, or shorthand that substitutes for actual content

If you write any such phrase, the receiving agent will see only that phrase — not the actual data — and will produce a completely wrong result. Always paste the real content directly.

## Core workflow
Follow the plan strictly, step by step. For each subtask in the plan, invoke the designated agent (with retries only if needed, see Execution rules), then move to the next subtask. After all subtasks have been processed, immediately produce a detailed summary of the entire execution process and output FINISHED. Do not revisit, re-evaluate, or re-execute any subtask that has already been completed.

## Tool call budget
Your total number of allowed tool calls is strictly limited — only slightly more than the number of subtasks in the plan. This budget is intentionally tight: it is enough to execute every subtask once with occasional retries, but not enough to repeat any subtask repeatedly or restart the entire plan. Excessive retries will consume the budget and prevent later subtasks from being executed at all.

## When invoking a tool, you must provide:
- The subtask content from the plan, passed as faithfully and completely as possible. Do not paraphrase, shorten, or omit details from the original subtask description. You may append clarifications, but never replace or reduce the original subtask content.
- All prior output and user-provided information that this subtask depends on, copied verbatim. No summaries. No references. The full actual text.

## Execution rules
- Execute subtasks in order, one at a time. Parallel tool calls are not allowed.
- Each subtask has a maximum of 3 attempts. Retrying is not required — only retry if you judge the result to be a clear failure or clearly insufficient. If the first attempt succeeds, move on immediately.
- If a subtask has been attempted 3 times and still fails, immediately write a detailed summary of the entire execution process so far — including what was attempted, what failed, and why it failed — then output FINISHED. Do not proceed to any remaining subtasks.
- Once all subtasks have been processed (either completed or exhausted their 3 attempts): write a detailed summary of every subtask's execution and result, then output FINISHED. Do this immediately — do not loop back, do not re-check, do not restart.

You never perform tasks yourself. You only control task scheduling, tool invocation, and dependency management.
"""

def get_one(
    model_client: OpenAIChatCompletionClient, callee_agents: List, **kwargs
) -> AssistantAgent:

    if "as_tool" in kwargs and kwargs["as_tool"]:
        system_message = caller_system_message2
        tools = []
        for agent in callee_agents:
            # tools.append(AgentTool(agent=agent, return_value_as_last_message=False))
            tools.append(AutoResetAgentTool(agent=agent, return_value_as_last_message=False))
    else:
        descriptions_for_other_agents = (
            "\n".join(f"{agent.name}: {agent.description}" for agent in callee_agents)
            + "\n"
        )
        system_message = (
            caller_system_message1.format(
                descriptions_for_other_agents=descriptions_for_other_agents
            ),
        )
        tools = None

    agent = AssistantAgent(
        name="Caller",
        description="An agent responsible for scheduling and invoking other agents to execute subtasks in sequence.",
        system_message=system_message,
        model_client=model_client,
        model_client_stream=False,
        tools=tools,
        max_tool_iterations=20,
        reflect_on_tool_use=True,
    )

    return agent

