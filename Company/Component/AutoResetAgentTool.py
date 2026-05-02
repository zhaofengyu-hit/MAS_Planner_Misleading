from typing import AsyncGenerator

from autogen_core import CancellationToken
from autogen_agentchat.tools import AgentTool
from autogen_agentchat.tools._task_runner_tool import TaskRunnerToolArgs
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


class AutoResetAgentTool(AgentTool):
    """AgentTool that resets the wrapped agent's conversation history before each invocation."""

    async def run(self, args: TaskRunnerToolArgs, cancellation_token: CancellationToken) -> TaskResult:
        await self._agent.on_reset(cancellation_token)
        return await super().run(args, cancellation_token)

    async def run_stream(
        self, args: TaskRunnerToolArgs, cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        await self._agent.on_reset(cancellation_token)
        async for event in super().run_stream(args, cancellation_token):
            yield event
