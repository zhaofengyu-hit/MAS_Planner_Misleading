"""
Download MCP Server

This module provides MCP server functionality for downloading files from URLs.
It supports HTTP/HTTPS downloads with configurable options and returns LLM-friendly formatted results.

Key features:
- Download files from HTTP/HTTPS URLs
- Configurable timeout
- Custom headers support for authentication
- LLM-optimized output formatting
- Comprehensive error handling and logging
- Path validation and directory creation

Main functions:
- mcp_download_file: Download files from URLs with comprehensive options
- mcp_get_download_capabilities: Get download service capabilities
"""

import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import re
import uuid
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

import os, sys

import urllib

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.append(project_dir)
from Company.Component.mcp_collections.base import (
    ActionArguments,
    ActionCollection,
    ActionResponse,
)


class DownloadResult(BaseModel):
    """Individual download operation result with structured data."""

    url: str
    file_path: str | None
    success: bool
    file_size: int | None = None
    duration: str
    timestamp: str
    error_message: str | None = None


class DownloadMetadata(BaseModel):
    """Metadata for download operation results."""

    url: str
    timeout_seconds: int
    output_path: str | None = None
    execution_time: float | None = None
    file_size_bytes: int | None = None
    content_type: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    headers_used: bool = False


class DownloadCollection(ActionCollection):
    """MCP service for file download operations with comprehensive controls.

    Provides secure file download capabilities including:
    - HTTP/HTTPS URL support
    - Configurable timeout controls
    - Custom headers for authentication
    - Path validation and directory creation
    - LLM-friendly result formatting
    - Error handling and logging
    """

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Configuration
        self.default_timeout = 60 * 3  # 3 minutes timeout
        self.max_file_size = 1024 * 1024 * 1024  # 1GB limit
        self.supported_schemes = {"http", "https"}

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_url(self, url: str) -> tuple[bool, str | None]:
        """Validate URL format and scheme.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            if not parsed.scheme:
                return False, "URL must include a scheme (http:// or https://)"

            if parsed.scheme.lower() not in self.supported_schemes:
                return (
                    False,
                    f"Unsupported URL scheme: {parsed.scheme}. Supported: {', '.join(self.supported_schemes)}",
                )

            if not parsed.netloc:
                return False, "URL must include a valid domain"

            return True, None

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _resolve_output_path(self, output_path: str) -> Path:
        """Resolve and validate output file path.

        Args:
            output_path: Output file path (absolute or relative)

        Returns:
            Resolved Path object
        """
        path = Path(output_path).expanduser()

        if not path.is_absolute():
            path = self.workspace / path

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        return path.resolve()

    def _format_download_output(
        self, result: DownloadResult, output_format: str = "markdown"
    ) -> str:
        """Format download results for LLM consumption.

        Args:
            result: Download execution result
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(result.model_dump(), indent=2)

        elif output_format == "text":
            output_parts = [
                f"URL: {result.url}",
                f"File Path: {result.file_path}",
                f"Status: {'SUCCESS' if result.success else 'FAILED'}",
                f"Duration: {result.duration}",
                f"Timestamp: {result.timestamp}",
            ]

            if result.file_size is not None:
                output_parts.append(f"File Size: {result.file_size:,} bytes")

            if result.error_message:
                output_parts.append(f"Error: {result.error_message}")

            return "\n".join(output_parts)

        else:  # markdown (default)
            status_emoji = "✅" if result.success else "❌"

            output_parts = [
                f"# File Download {status_emoji}",
                f"**URL:** `{result.url}`",
                f"**File Path:** `{result.file_path}`",
                f"**Status:** {'SUCCESS' if result.success else 'FAILED'}",
                f"**Duration:** {result.duration}",
                f"**Timestamp:** {result.timestamp}",
            ]

            if result.file_size is not None:
                size_mb = result.file_size / (1024 * 1024)
                output_parts.append(
                    f"**File Size:** {result.file_size:,} bytes ({size_mb:.2f} MB)"
                )

            if result.error_message:
                output_parts.extend(
                    ["\n## Error Details", f"```\n{result.error_message}\n```"]
                )

            return "\n".join(output_parts)

    def _sanitize_filename(self, name: str) -> str:
        name = Path(name).name
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        return name.strip() or "file"

    def _generate_unique_filename(self, output_dir: Path, suffix: str = "") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        while True:
            name = f"{uuid.uuid4().hex}{suffix}"
            path = output_dir / name
            if not path.exists():
                return path

    def _save_file(
        self,
        data_iter,
        output_dir: Path,
        original_filename: str | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if original_filename:
            filename = self._sanitize_filename(original_filename)
            output_path = output_dir / filename
        else:
            output_path = self._generate_unique_filename(output_dir)

        with open(output_path, "wb") as f:
            for chunk in data_iter:
                if chunk:
                    f.write(chunk)

        return output_path

    def _extract_original_filename(self, response, url: str) -> str | None:

        cd = response.headers.get("content-disposition")
        if cd:
            m = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I)
            if m:
                return urllib.parse.unquote(m.group(1))

            m = re.search(r'filename="?([^";]+)"?', cd, re.I)
            if m:
                return m.group(1)
            return None

        parsed = urllib.parse.urlparse(url)
        name = Path(parsed.path).name
        if name and "." in name:
            return name

        return None

    async def _download_file_async(
        self, url: str, output_path: Path, timeout: int, headers: dict[str, str] | None
    ) -> DownloadResult:
        """Download file asynchronously with comprehensive error handling.

        Args:
            url: URL to download from
            timeout: Request timeout in seconds
            headers: Optional custom headers

        Returns:
            DownloadResult with execution details
        """
        start_time = datetime.now()

        try:

            with requests.get(
                url, stream=True, timeout=timeout, headers=headers
            ) as response:
                response.raise_for_status()

                # Check content length if available
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(
                        f"File too large: {content_length} bytes (max: {self.max_file_size})"
                    )

                name = self._extract_original_filename(response, url)
                saved_path = self._save_file(
                    response.iter_content(chunk_size=8192), output_path, name
                )

                file_size = saved_path.stat().st_size
                duration = str(datetime.now() - start_time)

                return DownloadResult(
                    url=url,
                    file_path=str(saved_path.resolve()),
                    success=True,
                    file_size=file_size,
                    duration=duration,
                    timestamp=start_time.isoformat(),
                )

        except requests.exceptions.Timeout:
            duration = str(datetime.now() - start_time)
            error_msg = f"Download timed out after {timeout} seconds"

            return DownloadResult(
                url=url,
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

        except requests.exceptions.RequestException as e:
            duration = str(datetime.now() - start_time)
            error_msg = f"Request failed: {str(e)}"

            return DownloadResult(
                url=url,
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

        except Exception as e:
            duration = str(datetime.now() - start_time)
            error_msg = f"Unexpected error: {str(e)}"

            return DownloadResult(
                url=url,
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

    async def mcp_download_file(
        self,
        url: str = Field(description="HTTP/HTTPS URL of the file to download"),
        timeout: int = Field(
            default=60, description="Download timeout in seconds (default: 60)"
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text'",
        ),
    ) -> ActionResponse:
        """Download a file from a URL with comprehensive options and controls.

        This tool provides secure file download capabilities with:
        - HTTP/HTTPS URL support
        - Configurable timeout controls
        - Path validation and directory creation
        - File size limits and safety checks
        - LLM-optimized result formatting

        Args:
            url: The HTTP/HTTPS URL of the file to download
            timeout: Maximum download time in seconds
            output_format: Format for the response output

        Returns:
            ActionResponse with download results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(timeout, FieldInfo):
            timeout = timeout.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate URL
            url_valid, url_error = self._validate_url(url)
            if not url_valid:
                return ActionResponse(
                    success=False,
                    message=f"Invalid URL: {url_error}",
                    metadata=DownloadMetadata(
                        url=url,
                        timeout_seconds=timeout,
                        error_type="invalid_url",
                    ).model_dump(),
                )

            # Create output directory if it doesn't exist
            output_path = self.workspace / "downloads"
            output_path.mkdir(parents=True, exist_ok=True)

            # Perform download
            start_time = time.time()
            result = await self._download_file_async(
                url, output_path, timeout, self.headers
            )
            execution_time = time.time() - start_time

            # Format output
            formatted_output = self._format_download_output(result, output_format)

            # Create metadata
            metadata = DownloadMetadata(
                url=url,
                output_path=str(result.file_path),
                timeout_seconds=timeout,
                execution_time=execution_time,
                file_size_bytes=result.file_size,
                headers_used=self.headers is not None,
            )

            if not result.success:
                metadata.error_type = "download_failure"

            return ActionResponse(
                success=result.success,
                message=formatted_output,
                metadata=metadata.model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to download file: {str(e)}"
            self.logger.error(f"Download error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=DownloadMetadata(
                    url=url,
                    timeout_seconds=timeout,
                    error_type="internal_error",
                ).model_dump(),
            )

    def mcp_get_download_capabilities(self) -> ActionResponse:
        """Get information about download service capabilities and configuration.

        Returns:
            ActionResponse with download service capabilities and current configuration
        """
        capabilities = {
            "requests_available": requests is not None,
            "supported_schemes": list(self.supported_schemes),
            "supported_features": [
                "HTTP/HTTPS URL downloads",
                "Configurable timeout controls",
                "Custom headers support",
                "Path validation and directory creation",
                "File size limits and safety checks",
                "Multiple output formats (markdown, json, text)",
                "LLM-optimized result formatting",
                "Comprehensive error handling",
            ],
            "supported_formats": ["markdown", "json", "text"],
            "configuration": {
                "default_timeout": self.default_timeout,
                "max_file_size_bytes": self.max_file_size,
                "workspace": str(self.workspace),
            },
            "safety_features": [
                "URL validation",
                "File size limits",
                "Timeout controls",
                "Path validation",
                "Error handling and logging",
            ],
        }

        max_size_mb = self.max_file_size / (1024 * 1024)
        formatted_info = f"""# Download Service Capabilities

        ## Status
        - **Workspace:** `{self.workspace}`

        ## Supported Features
        {chr(10).join(f"- {feature}" for feature in capabilities["supported_features"])}

        ## Supported URL Schemes
        {chr(10).join(f"- {scheme}://" for scheme in capabilities["supported_schemes"])}

        ## Supported Output Formats
        {chr(10).join(f"- {fmt}" for fmt in capabilities["supported_formats"])}

        ## Configuration
        - **Default Timeout:** {capabilities["configuration"]["default_timeout"]} seconds
        - **Max File Size:** {self.max_file_size:,} bytes ({max_size_mb:.1f} MB)

        ## Safety Features
        {chr(10).join(f"- {feature}" for feature in capabilities["safety_features"])}
        """

        return ActionResponse(
            success=True,
            message=formatted_info,
            metadata=capabilities,
        )


# Default arguments for testing
if __name__ == "__main__":
    import os

    load_dotenv()

    arguments = ActionArguments(
        name="download",
        transport="stdio",
        workspace=os.getenv("MY_WORKSPACE", "~"),
    )
    try:
        service = DownloadCollection(arguments)
        service.run()
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
