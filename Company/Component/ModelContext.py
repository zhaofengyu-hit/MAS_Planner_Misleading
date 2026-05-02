import asyncio
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import UnboundedChatCompletionContext, BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, LLMMessage, ModelFamily
from autogen_ext.models.ollama import OllamaChatCompletionClient

class ReasoningModelContext(UnboundedChatCompletionContext):

    async def add_message(self, message):
        if isinstance(message, AssistantMessage):
            message.thought = None
        return await super().add_message(message)

    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                message.thought = None
            messages_out.append(message)
        return messages_out

class FilteredAndBufferedModelContext(BufferedChatCompletionContext):

    async def add_message(self, message):
        if isinstance(message, AssistantMessage):
            message.thought = None
        return await super().add_message(message)

    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                message.thought = None
            messages_out.append(message)
        return messages_out
