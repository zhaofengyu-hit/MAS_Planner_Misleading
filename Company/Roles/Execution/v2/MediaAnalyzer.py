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

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.append(project_dir)

mcp_timeout = 1200


def get_workbench_list() -> List[Workbench]:
    image_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/media/image.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    image_workbench = McpWorkbench(server_params=image_server_params)

    audio_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/media/audio.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    audio_workbench = McpWorkbench(server_params=audio_server_params)

    video_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/media/video.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    video_workbench = McpWorkbench(server_params=video_server_params)

    all_workbenches = [image_workbench, audio_workbench, video_workbench]
    return all_workbenches


NAME = "MediaAnalyzer"
DESCRIPTION = (
    "A multimedia analysis agent capable of handling local images, audio, and video. "
    "It provides OCR extraction, AI-based visual understanding, metadata inspection, audio transcription and trimming, video analysis and summarization, and key-frame extraction. "
    "Supports common formats across images (JPG/PNG/GIF/TIFF/WEBP …), audio (MP3/WAV/FLAC/AAC/OGG …), and video (MP4/AVI/MOV/MKV/WEBM …). "
    "GIF and video content are processed as multi-frame sequences—similar to video key-frame extraction—while static image formats are analyzed as single frames. "
    "Designed to perform assigned multimedia tasks efficiently and return clear, structured results."
)


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    workbenchs = get_workbench_list()

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message=f"""
You are the MediaAnalyzer agent.

Your responsibilities:
1. Process and analyze images, audio, and video using the tools available to you.
2. Use OCR, AI-based visual analysis, audio transcription, metadata extraction, video understanding, summarization, and key-frame extraction when appropriate.
3. Accept and operate on user-provided file paths or URLs. Only process supported multimedia formats.
4. Focus strictly on multimedia tasks. Do not perform unrelated reasoning or attempt actions outside your toolset.
5. Produce concise, accurate, and well-structured outputs.
6. When multiple tools could apply, choose the one that most directly completes the task.
7. 

Perform your assigned tasks carefully and efficiently.
""",
        model_client=model_client,
        workbench=workbenchs,
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=10,
    )

    return agent


