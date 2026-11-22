from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import AIModelEntity, FetchFrom
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
)

from models._modapi import (
    API_BASE,
    PRICING_PAGE,
    USER_AGENT,
    ModAPIError,
    api_key,
    check_key,
    find_model,
    model_tags,
    supports_openai_chat,
    usd_per_million,
)

DEFAULT_CONTEXT = 32768
DEFAULT_MAX_OUTPUT = 4096


class ModAPILargeLanguageModel(OAICompatLargeLanguageModel):
    """Chat models on ModAPI, which serves them through the OpenAI Chat Completions protocol."""

    def _modapi_credentials(self, credentials: Mapping[str, Any]) -> dict:
        try:
            key = api_key(credentials)
        except ModAPIError as exc:
            raise InvokeAuthorizationError(str(exc)) from exc
        prepared = dict(credentials)
        prepared.update(
            {
                "api_key": key,
                "endpoint_url": API_BASE,
                "mode": LLMMode.CHAT.value,
                "function_calling_type": "tool_call",
                "stream_function_calling": "supported",
                "extra_headers": {"User-Agent": USER_AGENT},
            }
        )
        return prepared

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = True,
        user: str | None = None,
    ) -> LLMResult | Generator:
        parameters = dict(model_parameters)
        if stream:
            # Ask for the real token counts in the last chunk, so Dify bills what ModAPI bills.
            parameters["stream_options"] = {"include_usage": True}
        return super()._invoke(
            model, self._modapi_credentials(credentials), prompt_messages, parameters, tools, stop, stream, user
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
    ) -> int:
        return super().get_num_tokens(model, self._modapi_credentials(credentials), prompt_messages, tools)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """For a model added by hand. Spends nothing: a free key check plus the public catalog."""
        try:
            check_key(credentials)
            entry = find_model(model)
        except ModAPIError as exc:
            raise CredentialsValidateFailedError(str(exc)) from exc
        if entry is None:
            raise CredentialsValidateFailedError(
                f"ModAPI does not list a model called {model}. Copy the exact model ID from {PRICING_PAGE}."
            )
        if not supports_openai_chat(entry):
            raise CredentialsValidateFailedError(
                f"{model} is not a chat model on ModAPI. Image, video and audio models are not available here."
            )

    def get_customizable_model_schema(self, model: str, credentials: Mapping) -> AIModelEntity | None:
        """Build the schema of a hand-added model from the public catalog and the limits the user entered."""
        try:
            entry = find_model(model) or {}
        except ModAPIError:
            entry = {}
        tags = model_tags(entry)
        context = int(credentials.get("context_size") or DEFAULT_CONTEXT)
        max_output = min(int(credentials.get("max_tokens") or DEFAULT_MAX_OUTPUT), context)
        features = []
        if "tools" in tags:
            features += ["tool-call", "multi-tool-call", "stream-tool-call", "agent-thought"]
        if "vision" in tags or "multimodal" in tags:
            features.append("vision")
        schema: dict[str, Any] = {
            "model": model,
            "label": {"en_US": model},
            "model_type": "llm",
            "fetch_from": FetchFrom.CUSTOMIZABLE_MODEL.value,
            "features": features,
            "model_properties": {"mode": "chat", "context_size": context},
            "parameter_rules": [
                {"name": "temperature", "use_template": "temperature"},
                {"name": "top_p", "use_template": "top_p"},
                {
                    "name": "max_tokens",
                    "use_template": "max_tokens",
                    "default": min(4096, max_output),
                    "min": 1,
                    "max": max_output,
                },
            ],
        }
        price = usd_per_million(entry) if entry else None
        if price:
            schema["pricing"] = {"input": price[0], "output": price[1], "unit": "0.000001", "currency": "USD"}
        return AIModelEntity.model_validate(schema)
