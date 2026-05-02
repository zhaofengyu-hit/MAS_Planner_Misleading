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
from pathlib import Path

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.append(project_dir)

mcp_timeout = 1200


def get_workbench_list() -> List[Workbench]:
    youtube_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/tools/youtube.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    youtube_workbench = McpWorkbench(server_params=youtube_server_params)

    arxiv_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/tools/mcparxiv.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    arxiv_workbench = McpWorkbench(server_params=arxiv_server_params)

    wiki_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/tools/wiki.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    wiki_workbench = McpWorkbench(server_params=wiki_server_params)

    download_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/tools/download.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    download_workbench = McpWorkbench(server_params=download_server_params)

    all_workbenches = [
        youtube_workbench,
        arxiv_workbench,
        wiki_workbench,
        download_workbench,
    ]
    return all_workbenches


playwright_wp = Path(__file__).parent.parent.parent.parent.parent.joinpath("tmp/browser_wp").absolute()

system_message = f"""
You are the WebSurfer agent. You retrieve information from the web by navigating and interacting with pages, exactly as a human would.

## Navigation policy
- Unless the user explicitly provides a URL or names a specific website to visit, always start at google.com. Open Google, type your search query, read the results, and follow links by clicking — do not jump directly to any URL, and do not construct or guess URLs from memory or training data.

## Page size management (critical)
Loading large, unfiltered pages makes it hard to locate relevant information and risks exceeding LLM context length limits. Before loading any page, actively try to reduce its size:
- Use the site's search box with specific keywords rather than browsing index pages.
- Apply any available filters: date range, category, author, language, file type, etc.
- Prefer paginated or scoped views over full listing pages.

## Tool selection
- Prefer specialized tools (ArXiv, Wikipedia, YouTube, direct download, etc.) when they directly match the task. Use browser automation only when no dedicated tool fits.
- Only Google Search is permitted as a search engine.
- For academic paper searches, prefer Google Scholar, arXiv, and DBLP.

## Output
- Produce well-structured, detailed output as required. Extract only what is relevant to the task.
- If you use browser tools to save any content, the browser will store it under the directory: [{playwright_wp}]. This is enforced by the browser itself — you only need to report the file path in your output when needed.

Work carefully, efficiently, and stay within your assigned retrieval scope.
"""

NAME = "WebSurfer"
DESCRIPTION = (
    "A web-focused agent capable of browser-based navigation and a range of online retrieval tasks. "
    "Key capabilities include: controlling a browser to visit and extract information from web pages; searching and downloading academic papers from arXiv; looking up content on Wikipedia; downloading YouTube videos (with automatic audio extraction); and general-purpose file downloading from URLs. "
    "It can be used to extract textual information from web pages, which can then be passed for further tasks."
)


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    if "url" in kwargs:
        server_params = SseServerParams(
            url=kwargs["url"], timeout=60, sse_read_timeout=60 * 10
        )
    else:
        if not playwright_wp.exists():
            playwright_wp.mkdir(parents=True, exist_ok=True)
        server_params = StdioServerParams(
            command="npx",
            args=[
                "--yes",
                "@playwright/mcp@latest",
                "--image-responses",
                "omit",
            ],
            cwd=str(playwright_wp),
            read_timeout_seconds=mcp_timeout,
        )
    workbenchs = get_workbench_list()
    workbenchs.append(McpWorkbench(server_params=server_params))

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message=system_message,
        model_client=model_client,
        workbench=workbenchs,
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=20,
    )

    return agent

