from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model import AIModelEntity, EmbeddingInputType, FetchFrom
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
)

from models._modapi import (
    API_BASE,
    PRICING_PAGE,
    ModAPIError,
    api_key,
    check_key,
    find_model,
    supports_embeddings,
)

DEFAULT_CONTEXT = 8191


class ModAPITextEmbeddingModel(OAICompatEmbeddingModel):
    """Embedding models on ModAPI through the OpenAI /v1/embeddings protocol."""

    def _modapi_credentials(self, credentials: Mapping[str, Any]) -> dict:
        try:
            key = api_key(credentials)
        except ModAPIError as exc:
            raise InvokeAuthorizationError(str(exc)) from exc
        prepared = dict(credentials)
        prepared.update({"api_key": key, "endpoint_url": API_BASE})
        return prepared

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: str | None = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        return super()._invoke(model, self._modapi_credentials(credentials), texts, user, input_type)

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        return super().get_num_tokens(model, self._modapi_credentials(credentials), texts)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            check_key(credentials)
            entry = find_model(model)
        except ModAPIError as exc:
            raise CredentialsValidateFailedError(str(exc)) from exc
        if entry is None or not supports_embeddings(entry):
            raise CredentialsValidateFailedError(
                f"ModAPI does not list an embedding model called {model}. See {PRICING_PAGE}."
            )

    def get_customizable_model_schema(self, model: str, credentials: Mapping) -> AIModelEntity | None:
        context = int(credentials.get("context_size") or DEFAULT_CONTEXT)
        return AIModelEntity.model_validate(
            {
                "model": model,
                "label": {"en_US": model},
                "model_type": "text-embedding",
                "fetch_from": FetchFrom.CUSTOMIZABLE_MODEL.value,
                "model_properties": {"context_size": context, "max_chunks": 32},
            }
        )
