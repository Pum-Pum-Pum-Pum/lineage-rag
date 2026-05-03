from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.core.config import get_settings
from app.llm.prompt_template import GroundedPrompt
from app.llm.usage import LLMUsage


@dataclass(frozen=True)
class LLMCompletionResult:
    content: str
    usage: LLMUsage


def get_llm_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )


def generate_chat_completion(
    prompt: GroundedPrompt,
    model: str | None = None,
    client: Any | None = None,
) -> str:
    """Generate one chat completion from a grounded prompt."""

    return generate_chat_completion_with_usage(
        prompt=prompt,
        model=model,
        client=client,
    ).content


def generate_chat_completion_with_usage(
    prompt: GroundedPrompt,
    model: str | None = None,
    client: Any | None = None,
) -> LLMCompletionResult:
    """Generate one chat completion and return text plus usage metadata."""

    settings = get_settings()
    llm_client = client or get_llm_client()
    selected_model = model or settings.openai_chat_model

    response = llm_client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
    )

    if not response.choices:
        raise RuntimeError("LLM response did not contain any choices.")

    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise RuntimeError("LLM response content was empty.")

    return LLMCompletionResult(
        content=content.strip(),
        usage=extract_llm_usage(response, selected_model),
    )


def extract_llm_usage(response: Any, model: str) -> LLMUsage:
    """Extract token usage from an OpenAI-compatible chat response."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage(
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )

    return LLMUsage(
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
