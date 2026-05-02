"""LLM client wrapper for API calls."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import json
import openai
from openai import OpenAI
import os

from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.model = model
        self.temperature = None if model == "gpt-5" else temperature
        self.max_tokens = max_tokens

        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def ask(self, user_input: list[str]) -> str:
        for item in user_input:
            self.messages.append({"role": "user", "content": item})

        response = self.client.responses.create(
            model=self.model,
            input=self.messages,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        answer = response.output_text
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def ask_structured(
        self,
        user_input: list[str],
        model_cls: Type[T],
    ) -> T:

        if self.model.startswith("qwen"):
            return self.ask_structured_qwen(
                user_input=user_input,
                model_cls=model_cls,
            )

        for item in user_input:
            self.messages.append({"role": "user", "content": item})

        response = self.client.responses.parse(
            model=self.model,
            input=self.messages,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            text_format=model_cls,
        )

        parsed = response.output_parsed
        result = model_cls.model_validate(parsed)

        self.messages.append({"role": "assistant", "content": result.model_dump_json()})

        return result

    def ask_structured_qwen(
        self,
        user_input: list[str],
        model_cls: Type[T],
    ) -> T:
        """Structured output via chat.completions.create; compatible with DeepSeek and other OpenAI-compatible APIs."""
        for item in user_input:
            self.messages.append({"role": "user", "content": item})

        json_schema = model_cls.model_json_schema()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": model_cls.__name__,
                    "schema": json_schema,
                },
                "strict": True,
            },
            # extra_body={"enable_thinking": True},
        )

        content = response.choices[0].message.content
        result = model_cls.model_validate_json(content)

        self.messages.append({"role": "assistant", "content": content})

        return result

    def reset(self, system_prompt: Optional[str] = None):

        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

