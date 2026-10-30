"""Builds the configured LLM provider."""

from __future__ import annotations

import logging

from agent_backend.config import Settings
from agent_backend.llm.base import LLMProvider
from agent_backend.llm.fake import FakeLLM

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMProvider:
    provider = settings.resolved_llm_provider()
    if provider == "anthropic":
        from agent_backend.llm.anthropic_provider import AnthropicProvider

        llm: LLMProvider = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_output_tokens=settings.llm_max_output_tokens,
            effort=settings.anthropic_effort,
            refusal_fallbacks=settings.anthropic_refusal_fallbacks,
        )
    elif provider == "openai":
        from agent_backend.llm.openai_provider import OpenAIProvider

        llm = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    else:
        llm = FakeLLM(token_delay_s=settings.fake_llm_token_delay_ms / 1000)
    logger.info("LLM provider ready", extra={"provider": llm.name})
    return llm
