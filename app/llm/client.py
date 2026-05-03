from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.core.config import get_settings
from app.llm.prompt_template import GroundedPrompt


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

    return content.strip()
