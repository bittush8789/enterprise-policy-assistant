from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from .config import normalize_model_name, settings


def get_chat_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a Groq chat language model with the configured or requested model."""
    resolved_model = normalize_model_name(model or settings.groq_model)
    key = (
        api_key
        or kwargs.pop("groq_api_key", None)
        or settings.groq_api_key
        or os.getenv("GROQ_API_KEY")
    )

    if not key or key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not configured. Please set your GROQ_API_KEY in the .env file."
        )

    return ChatGroq(
        model=resolved_model,
        temperature=temperature,
        api_key=key,
        **kwargs,
    )
