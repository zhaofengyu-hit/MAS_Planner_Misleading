"""
YouTube MCP Service

This module provides MCP service functionality for YouTube operations including:
- Downloading videos from YouTube URLs
- Extracting transcripts from YouTube videos

It handles various scenarios with proper validation, error handling,
and progress tracking while providing LLM-friendly formatted results.
"""

import os, sys
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


from yt_dlp import YoutubeDL

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
sys.path.append(project_dir)
from Company.Component.mcp_collections.base import (
    ActionArguments,
    ActionCollection,
    ActionResponse,
)

# Default driver path for Chrome WebDriver
_DEFAULT_DRIVER_PATH = os.environ.get(
    "CHROME_DRIVER_PATH",
    str(Path("~/Downloads/chromedriver-linux64/chromedriver").expanduser()),
)


class YoutubeDownloadResults(BaseModel):
    """Download result model with file information"""

    video_file_path: str
    video_file_name: str
    video_file_size: int
    video_content_type: str | None=None
    extracted_audio_file_path: str
    extracted_audio_file_name: str
    extracted_audio_file_size: int
    extracted_audio_content_type: str | None=None
    success: bool
    error: str | None = None


class YouTubeMetadata(BaseModel):
    """Metadata for YouTube operation results"""

    operation: str
    url: str | None = None
    video_id: str | None = None
    video_file_path: str | None = None
    video_file_name: str | None = None
    video_file_size: int | None = None
    video_content_type: str | None=None
    extracted_audio_file_path: str | None = None
    extracted_audio_file_name: str | None = None
    extracted_audio_file_size: int | None = None
    extracted_audio_content_type: str | None=None
    language_code: str | None = None
    translate_to_language: str | None = None
    execution_time: float | None = None
    error_type: str | None = None


class YouTubeActionCollection(ActionCollection):
    """MCP service for YouTube operations.

    Provides YouTube capabilities including:
    - Video downloading with Selenium automation
    - Transcript extraction and translation
    - LLM-friendly result formatting
    - Error handling and logging
    """

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Initialize supported file extensions
        self.supported_extensions = {".mp4", ".webm", ".mkv"}


    def _format_download_output(
        self, result: YoutubeDownloadResults, format_type: str = "markdown"
    ) -> str:
        """Format download results for LLM consumption.

        Args:
            result: Download result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to download video: {result.error}"

        if format_type == "json":
            return result.model_dump()
        elif format_type == "text":
            output_parts = [
                "Download completed successfully",
                f"Video File: {result.video_file_name}",
                f"Video Path: {result.video_file_path}",
                f"Video Size: {result.video_file_size} bytes",
                f"Extracted Audio File: {result.extracted_audio_file_name}",
                f"Extracted Audio Path: {result.extracted_audio_file_path}",
                f"Extracted Audio Size: {result.extracted_audio_file_size} bytes",
            ]
            if result.video_content_type:
                output_parts.append(f"Video Content Type: {result.video_content_type}")
            if result.extracted_audio_content_type:
                output_parts.append(
                    f"Extracted Audio Content Type: {result.extracted_audio_content_type}"
                )

            return "\n".join(output_parts)
        else:  # markdown (default)
            output_parts = [
                "# YouTube Download Results ✅",
                "",
                "## File Information",
                f"**Video Filename:** `{result.video_file_name}`",
                f"**Video Path:** `{result.video_file_path}`",
                f"**Video Size:** {result.video_file_size} bytes",
                f"**Extracted Audio Filename:** `{result.extracted_audio_file_name}`",
                f"**Extracted Audio Path:** `{result.extracted_audio_file_path}`",
                f"**Extracted Audio Size:** {result.extracted_audio_file_size} bytes",
            ]
            if result.video_content_type:
                output_parts.append(
                    f"**Video Content Type:** {result.video_content_type}"
                )
            if result.extracted_audio_content_type:
                output_parts.append(
                    f"**Extracted Audio Content Type:** {result.extracted_audio_content_type}"
                )

            return "\n".join(output_parts)

    def _extract_mp3(self,input_video, output_mp3):
        input_video = Path(input_video).resolve()
        output_mp3 = Path(output_mp3).resolve()

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{input_video.parent}:/input",
            "-v", f"{output_mp3.parent}:/output",
            "linuxserver/ffmpeg",
            "-i", f"/input/{input_video.name}",
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "320k",
            "-y",
            f"/output/{output_mp3.name}",
        ]

        subprocess.run(cmd, check=True)

    def _download_with_yt_dlp(self, url: str, output_dir: str) -> tuple:
        """Download YouTube video and extract MP3 audio using yt-dlp.
        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded content

        Returns:
            (video_path, audio_path)
        """

        ydl_opts = {
            "format": "mp4",
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "keepvideo": True,
            # "postprocessors": [
            #     {
            #         "key": "FFmpegExtractAudio",
            #         "preferredcodec": "mp3",
            #         "preferredquality": "320",
            #     },
            # ],
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id")

            video_path = os.path.join(output_dir, f"{video_id}.mp4")
            audio_path = os.path.join(output_dir, f"{video_id}.mp3")

        self._extract_mp3(video_path, audio_path)
        return (Path(video_path), Path(audio_path))

    def _get_youtube_content(self, url: str, output_dir: str, timeout: int) -> None:
        """Use Selenium to download YouTube content via cobalt.tools

        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded content
            timeout: Maximum time to wait for download in seconds
        """
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            # Set download file default path
            prefs = {
                "download.default_directory": output_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            }
            options.add_experimental_option("prefs", prefs)
            options.add_argument("--headless=new")

            # Create WebDriver object and launch Chrome browser
            service = Service(executable_path=_DEFAULT_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)

            # Open target webpage
            # driver.get("https://cobalt.tools/") # doesn't work now
            driver.get("https://ytdown.to/")  # does't work now
            # https://yt1s.com.co/ works now

            # Wait for page to load
            time.sleep(5)
            # Find input field and enter YouTube link
            # input_field = driver.find_element(By.ID, "link-area")
            input_field = driver.find_element(By.NAME, "URLz")
            input_field.send_keys(url)
            time.sleep(5)
            # Find download button and click
            # download_button = driver.find_element(By.ID, "download-button")
            download_button = driver.find_element(
                By.CSS_SELECTOR,
                "#ytdown-downloader-form > div.input-group > button.btn-download",
            )
            download_button.click()
            time.sleep(5)

            # when the download link appears, click it
            download_video_link = driver.find_element(
                By.CSS_SELECTOR, "#downloadButton"
            )
            download_video_link.click()
            time.sleep(5)

            cnt = 0
            while (
                len(os.listdir(output_dir)) == 0
                or os.listdir(output_dir)[0].split(".")[-1] == "crdownload"
            ):
                time.sleep(3)
                cnt += 3
                if cnt >= timeout:
                    break

        except Exception as e:
            raise
        finally:
            # Close browser
            if driver:
                driver.quit()

    def _find_existing_video(self, search_dir: str, video_id: str) -> str | None:
        """Recursively search for an existing video file with the given ID.

        Args:
            search_dir: Directory to search in
            video_id: YouTube video ID to look for

        Returns:
            Path to existing file if found, None otherwise
        """
        if not video_id:
            return None

        search_path = Path(search_dir)
        if not search_path.exists():
            return None

        for item in search_path.iterdir():
            if item.is_file() and video_id in item.name:
                return str(item)
            elif item.is_dir():
                found = self._find_existing_video(str(item), video_id)
                if found:
                    return found

        return None

    async def mcp_download_youtube_video(
        self,
        url: str = Field(description="The URL of YouTube video to download."),
        timeout: int = Field(
            600, description="Download timeout in seconds (default: 600)."
        ),
        output_format: str = Field(
            "markdown",
            description="Output format: 'markdown', 'json', or 'text' (default: markdown).",
        ),
    ) -> ActionResponse:
        """Download a YouTube video from URL and save it to the local filesystem.

        This tool provides YouTube video downloading with:
        - Selenium-based automation via cobalt.tools
        - Configurable timeout controls
        - Existing file detection to avoid redundant downloads
        - Automatic audio extraction: after downloading the video, the audio track is separated and saved as an MP3 file
        - LLM-optimized result formatting

        Args:
            url: The URL of YouTube video to download
            timeout: Maximum download time in seconds
            output_format: Format for the response output

        Returns:
            ActionResponse with download results and metadata
        """
        start_time = time.time()

        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    "Invalid URL format. URL must start with http:// or https://"
                )

            if not ("youtube.com" in url or "youtu.be" in url):
                raise ValueError("URL must be a valid YouTube URL")

            # Create output directory if it doesn't exist
            output_path = self.workspace / "youtube_downloads"
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename based on timestamp
            filename = f"youtube_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_path = output_path / filename
            file_path.mkdir(parents=True, exist_ok=True)

            # Extract video ID for existing file check
            video_id = url.split("?v=")[-1].split("&")[0] if "?v=" in url else ""
            if "youtu.be/" in url and not video_id:
                video_id = url.split("youtu.be/")[-1].split("?")[0]

            # Check if video already exists
            base_path = self.workspace
            # existing_file = self._find_existing_video(str(base_path), video_id)
            existing_file = self._find_existing_video(str(output_path), video_id+".mp4")

            if existing_file:
                video_path = Path(existing_file)
                audio_path = video_path.with_suffix(".mp3")
                result = YoutubeDownloadResults(
                    video_file_path=str(video_path),
                    video_file_name=video_path.name,
                    video_file_size=video_path.stat().st_size,
                    video_content_type="mp4",
                    extracted_audio_file_path=str(audio_path),
                    extracted_audio_file_name=audio_path.name,
                    extracted_audio_file_size=audio_path.stat().st_size,
                    extracted_audio_content_type="mp3",
                    success=True,
                    error=None,
                )

                # Format output for LLM
                message = self._format_download_output(result, output_format)
                execution_time = time.time() - start_time

                # Create metadata
                metadata = YouTubeMetadata(
                    operation="download",
                    url=url,
                    video_id=video_id,
                    video_file_path=str(video_path),
                    video_file_name=video_path.name,
                    video_file_size=video_path.stat().st_size,
                    video_content_type="mp4",
                    extracted_audio_file_path=str(audio_path),
                    extracted_audio_file_name=audio_path.name,
                    extracted_audio_file_size=audio_path.stat().st_size,
                    extracted_audio_content_type="mp3",
                    execution_time=execution_time,
                ).model_dump()

                return ActionResponse(success=True, message=message, metadata=metadata)

            # Download the video
            # self._get_youtube_content(url, str(file_path), timeout)
            video_path, audio_path = self._download_with_yt_dlp(url, str(file_path))

            if not video_path.exists() or not audio_path.exists():
                raise FileNotFoundError("No files were downloaded")

            # Create result
            result = YoutubeDownloadResults(
                video_file_path=str(video_path),
                video_file_name=video_path.name,
                video_file_size=video_path.stat().st_size,
                video_content_type="mp4",
                extracted_audio_file_path=str(audio_path),
                extracted_audio_file_name=audio_path.name,
                extracted_audio_file_size=audio_path.stat().st_size,
                extracted_audio_content_type="mp3",
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_download_output(result, output_format)
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="download",
                url=url,
                video_id=video_id,
                video_file_path=str(video_path),
                video_file_name=video_path.name,
                video_file_size=video_path.stat().st_size,
                video_content_type="mp4",
                extracted_audio_file_path=str(audio_path),
                extracted_audio_file_name=audio_path.name,
                extracted_audio_file_size=audio_path.stat().st_size,
                extracted_audio_content_type="mp3",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            error_msg = str(e)

            # Format error for LLM
            message = f"Failed to download YouTube video: {error_msg}"
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="download",
                url=url,
                error_type="download_failure",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=False, message=message, metadata=metadata)


# Default arguments for testing
if __name__ == "__main__":
    load_dotenv()

    arguments = ActionArguments(
        name="youtube_service",
        transport="stdio",
        workspace=os.getenv("MY_WORKSPACE", "~"),
    )

    try:
        youtube_service = YouTubeActionCollection(arguments)
        youtube_service.run()
    except Exception as e:
        print(f"Error: {e}")
