import logging
import json
import time

from autogen_core.logging import LLMCallEvent
from autogen_core import EVENT_LOGGER_NAME


class LLMUsageTracker(logging.Handler):
    def __init__(self) -> None:
        """Logging handler that tracks the number of tokens used in the prompt and completion."""
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._events = []

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._events = []

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record. To be used by the logging module."""
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens
                self._events.append(event)
        except Exception:
            self.handleError(record)

class LLMEventLogger:
    def __init__(self):
        self._logger = logging.getLogger(EVENT_LOGGER_NAME)
        self._logger.setLevel(logging.INFO)

        self._usage_tracker = LLMUsageTracker()
        self._start_time = None
        self._end_time = None
        self._attached = False

    def start(self):
        """Attach handlers and reset state."""
        if not self._attached:
            self._start_time = time.time()
            self._logger.handlers = []
            self._logger.addHandler(self._usage_tracker)
            self._attached = True

        self._usage_tracker.reset()

    def stop(self):
        """Detach handlers if needed."""
        if self._attached:
            self._end_time = time.time()
            self._logger.removeHandler(self._usage_tracker)
            self._attached = False

    def reset_tracker(self):
        """Reset usage stats."""
        self._usage_tracker.reset()

    def save(self, path: str):
        """Save usage stats to file."""
        data = [event.kwargs for event in self._usage_tracker._events]
        extra_data = {
            "start_time": self._start_time,
            "end_time": self._end_time,
        }
        data.append(extra_data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )