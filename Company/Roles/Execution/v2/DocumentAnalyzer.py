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
# from Company.Component.mcp_collections.documents import mscsv, msdocx, mspptx, pdf, txt

import Company.Component.FileParserWorkbench as FileParserWorkbench

mcp_timeout = 1200


def get_workbench_list() -> List[Workbench]:
    csv_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/documents/mscsv.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    csv_workbench = McpWorkbench(server_params=csv_server_params)

    docx_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/documents/msdocx.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    docx_workbench = McpWorkbench(server_params=docx_server_params)

    pptx_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/documents/mspptx.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    pptx_workbench = McpWorkbench(server_params=pptx_server_params)

    xlsx_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/documents/msxlsx.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    xlsx_workbench = McpWorkbench(server_params=xlsx_server_params)

    plain_txt_server_params = StdioServerParams(
        command="python",
        args=[
            "Company/Component/mcp_collections/documents/txt.py",
        ],
        read_timeout_seconds=mcp_timeout,
    )
    plain_txt_workbench = McpWorkbench(server_params=plain_txt_server_params)

    all_workbenches = [
        csv_workbench,
        docx_workbench,
        pptx_workbench,
        xlsx_workbench,
        plain_txt_workbench,
        FileParserWorkbench.get_workbench(),
    ]
    return all_workbenches


NAME = "DocumentAnalyzer"
DESCRIPTION = (
    "A comprehensive document-processing agent capable of extracting, analyzing, and structuring content from a wide range of local office files and text formats. "
    "Supports CSV/TSV, Word (DOC/DOCX), PowerPoint (PPT/PPTX), Excel (XLS/XLSX), PDF, and diverse text/code files. "
    "Provides unified text extraction, table and media extraction, metadata collection, structural analysis, and multi-format output (Markdown/JSON/HTML/Text). "
    "Designed to deliver clean, organized results for downstream reasoning and further processing. "
    "It can extract embedded media as raw files or generate screenshots for visual spreadsheet elements, which are then passed to specialized agents for further analysis."
)


def get_one(model_client: OpenAIChatCompletionClient, description: str | None = None, **kwargs) -> AssistantAgent:

    workbenchs = get_workbench_list()

    agent = AssistantAgent(
        name=NAME,
        description=description or DESCRIPTION,
        system_message=f"""
You are the DocumentAnalyzer agent.

Your responsibilities:
1. Extract and analyze content from documents using your available tools.
2. Support CSV/TSV, Word, PowerPoint, Excel, PDF, and general text/code files, operating only on supported formats.
3. Provide structured outputs such as text, tables, media summaries, metadata, and document structure when applicable.
4. Stay strictly within document-processing tasks. Do not perform unrelated reasoning or actions outside your toolset.
5. Choose the most appropriate extraction tool based on file type and user intent.
6. Produce clear, concise, and well-formatted results suitable for further LLM processing.
7. If the file is too large and the task cannot be completed successfully, upload the file to RAGFlow and retrieve the required content from it, which is usually successful.

Perform all assigned tasks carefully and reliably.
""",
        model_client=model_client,
        workbench=workbenchs,
        reflect_on_tool_use=True,
        model_client_stream=False,
        max_tool_iterations=100,
    )

    return agent

