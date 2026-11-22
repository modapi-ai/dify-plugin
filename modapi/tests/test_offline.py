"""Offline tests: load the plugin the way the Dify runtime does and mock every HTTP call."""

import json
import os
from pathlib import Path

import pytest
import requests_mock as rm
import yaml

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
API = "https://modapi.ai/v1"
CATALOG = "https://modapi.ai/api/pricing"
CATALOG_BODY = {
    "data": [
        {
            "model_name": "gpt-4.1-nano",
            "quota_type": 0,
            "model_ratio": 0.05,
            "completion_ratio": 4,
            "tags": "chat,tools,vision",
            "supported_endpoint_types": ["openai"],
            "enable_groups": ["LLM_Standard"],
        },
        {
            "model_name": "kimi-k3",
            "quota_type": 0,
            "model_ratio": 1.5,
            "completion_ratio": 5,
            "tags": "chat,reasoning,tools,vision",
            "supported_endpoint_types": ["openai"],
            "enable_groups": ["LLM_Standard"],
        },
        {
            "model_name": "text-embedding-3-small",
            "quota_type": 0,
            "model_ratio": 0.01,
            "completion_ratio": 1,
            "tags": "text-embeddings",
            "supported_endpoint_types": ["openai", "embedding"],
            "enable_groups": ["LLM_Standard"],
        },
        {
            "model_name": "kling-v3",
            "quota_type": 1,
            "tags": "video",
            "supported_endpoint_types": ["video-generations"],
            "enable_groups": ["Multimedia"],
        },
    ]
}
CREDS = {"api_key": "sk-test"}


@pytest.fixture(scope="module")
def registration():
    from dify_plugin import DifyPluginEnv
    from dify_plugin.core.plugin_registration import PluginRegistration

    cwd = os.getcwd()
    os.chdir(PLUGIN_ROOT)
    try:
        yield PluginRegistration(DifyPluginEnv())
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def fresh_catalog_cache():
    from models import _modapi

    _modapi._catalog_cache.update(at=0.0, models={})


def _model(registration, model_type):
    from dify_plugin.entities.model import ModelType

    _config, _provider, factory = registration.models_mapping["modapi"]
    return factory.get_instance(ModelType(model_type))


def test_plugin_loads_with_predefined_models(registration):
    config = registration.models_configuration[0]
    assert config.provider == "modapi"
    by_type = {}
    for m in config.models:
        by_type.setdefault(m.model_type.value, []).append(m.model)
    assert "gpt-4.1-nano" in by_type["llm"]
    assert "text-embedding-3-small" in by_type["text-embedding"]
    assert len(by_type["llm"]) >= 20


def test_every_model_yaml_has_limits_and_price():
    for path in sorted((PLUGIN_ROOT / "models").glob("*/*.yaml")):
        if path.name == "_position.yaml":
            continue
        data = yaml.safe_load(path.read_text())
        assert data["model_properties"]["context_size"] > 0, path.name
        assert float(data["pricing"]["input"]) > 0, path.name
        if data["model_type"] == "llm":
            out_rule = next(r for r in data["parameter_rules"] if r.get("use_template") == "max_tokens")
            if "max" in out_rule:
                assert 0 < out_rule["max"] <= data["model_properties"]["context_size"], path.name


def test_engineering_parameters_are_applied():
    def load(name):
        return yaml.safe_load((PLUGIN_ROOT / "models" / "llm" / f"{name}.yaml").read_text())

    def out_rule(data):
        return next(r for r in data["parameter_rules"] if r.get("use_template") == "max_tokens")

    for name in ("kimi-k3", "minimax-m3-1-flash"):
        assert out_rule(load(name))["name"] == "max_completion_tokens", name
    # Max output unconfirmed: not preset, users add them by ID.
    for held in ("grok-4", "deepseek-v3-1", "qwq-32b"):
        assert not (PLUGIN_ROOT / "models" / "llm" / f"{held}.yaml").exists(), held
    assert load("deepseek-v4-1-flash")["model_properties"]["context_size"] == 1000000
    assert out_rule(load("deepseek-v4-1-flash"))["max"] == 393216
    # Vendor K/M values rounded down to decimal integers (K = 1,000, M = 1,000,000).
    expected = {"glm-5-3-flash": (1000000, 128000), "glm-4-5": (128000, 96000), "glm-4-airx": (8000, 4000),
                "glm-4-flash": (128000, 4000), "glm-4-long": (1000000, 4000)}
    for name, (context, max_out) in expected.items():
        data = load(name)
        assert data["model_properties"]["context_size"] == context, name
        assert out_rule(data)["max"] == max_out, name


def test_position_files_list_every_model():
    for kind in ("llm", "text_embedding"):
        folder = PLUGIN_ROOT / "models" / kind
        listed = yaml.safe_load((folder / "_position.yaml").read_text())
        shipped = {yaml.safe_load(p.read_text())["model"] for p in folder.glob("*.yaml") if p.name != "_position.yaml"}
        assert set(listed) == shipped


def test_manifest_meets_marketplace_rules():
    manifest = yaml.safe_load((PLUGIN_ROOT / "manifest.yaml").read_text())
    assert manifest["tags"] and set(manifest["tags"]) <= {"llm", "model_provider", "productivity"}
    assert manifest["contact"] and manifest["privacy"] == "PRIVACY.md"
    assert (PLUGIN_ROOT / "PRIVACY.md").exists() and (PLUGIN_ROOT / "_assets" / manifest["icon"]).exists()
    assert manifest["network"]["domains"] == ["modapi.ai"]


def test_provider_key_check_uses_free_model_list(registration):
    _config, provider, _factory = registration.models_mapping["modapi"]
    with rm.Mocker() as m:
        m.get(f"{API}/models", json={"data": []})
        provider.validate_provider_credentials(CREDS)
        assert m.call_count == 1
        assert m.last_request.headers["Authorization"] == "Bearer sk-test"
        assert m.last_request.method == "GET"


def test_provider_rejects_bad_key(registration):
    from dify_plugin.errors.model import CredentialsValidateFailedError

    _config, provider, _factory = registration.models_mapping["modapi"]
    with rm.Mocker() as m:
        m.get(f"{API}/models", status_code=401, json={"error": {"code": "token_invalid"}})
        with pytest.raises(CredentialsValidateFailedError, match="rejected this API key"):
            provider.validate_provider_credentials({"api_key": "bad"})


def test_chat_stream_sends_openai_request_and_reads_usage(registration):
    from dify_plugin.entities.model.message import UserPromptMessage

    llm = _model(registration, "llm")
    chunks = [
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        },
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [{"index": 0, "delta": {"content": " from ModAPI"}, "finish_reason": "stop"}],
        },
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-nano",
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    with rm.Mocker() as m:
        m.post(f"{API}/chat/completions", text=body, headers={"Content-Type": "text/event-stream"})
        result = llm.invoke(
            model="gpt-4.1-nano",
            credentials=dict(CREDS),
            prompt_messages=[UserPromptMessage(content="Say hello")],
            model_parameters={"max_tokens": 64},
            stream=True,
        )
        out = list(result)
        sent = m.last_request.json()
    text = "".join(c.delta.message.content for c in out if isinstance(c.delta.message.content, str))
    assert text == "Hello from ModAPI"
    assert sent["model"] == "gpt-4.1-nano"
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 64
    assert m.last_request.headers["Authorization"] == "Bearer sk-test"
    usage = out[-1].delta.usage
    assert usage.prompt_tokens == 7 and usage.completion_tokens == 4


def test_custom_model_is_checked_against_catalog(registration):
    from dify_plugin.errors.model import CredentialsValidateFailedError

    llm = _model(registration, "llm")
    with rm.Mocker() as m:
        m.get(f"{API}/models", json={"data": []})
        m.get(CATALOG, json=CATALOG_BODY)
        llm.validate_credentials("kimi-k3", dict(CREDS))
        with pytest.raises(CredentialsValidateFailedError, match="does not list"):
            llm.validate_credentials("no-such-model", dict(CREDS))
        with pytest.raises(CredentialsValidateFailedError, match="not a chat model"):
            llm.validate_credentials("kling-v3", dict(CREDS))
        assert not any(r.method == "POST" for r in m.request_history), "validation must not send paid requests"


def test_custom_model_schema_uses_catalog_price_and_user_limits(registration):
    llm = _model(registration, "llm")
    with rm.Mocker() as m:
        m.get(CATALOG, json=CATALOG_BODY)
        schema = llm.get_customizable_model_schema(
            "kimi-k3", {**CREDS, "context_size": "262144", "max_tokens": "16384"}
        )
    from dify_plugin.entities.model import ModelPropertyKey

    assert schema.model_properties[ModelPropertyKey.CONTEXT_SIZE] == 262144
    assert str(schema.pricing.input) == "3" and str(schema.pricing.output) == "15"
    features = {f.value for f in schema.features}
    assert {"tool-call", "vision"} <= features
    max_rule = next(r for r in schema.parameter_rules if r.name == "max_tokens")
    assert max_rule.max == 16384


def test_embeddings_request(registration, monkeypatch):
    embedding = _model(registration, "text-embedding")
    # The SDK counts tokens with a GPT-2 tokenizer it downloads on first use; keep the test offline.
    monkeypatch.setattr(embedding, "_get_num_tokens_by_gpt2", lambda text: len(text.split()))
    with rm.Mocker() as m:
        m.post(
            f"{API}/embeddings",
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )
        result = embedding.invoke(model="text-embedding-3-small", credentials=dict(CREDS), texts=["hello"])
        sent = m.last_request.json()
    assert sent["model"] == "text-embedding-3-small" and sent["input"] == ["hello"]
    assert len(result.embeddings) == 1
