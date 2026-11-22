"""Generate the predefined model YAML files of the ModAPI Dify plugin.

Usage:
  python3 -I generate_models.py <plugin dir> <pricing.json> <litellm model_prices_and_context_window.json>

Sources:
  * https://modapi.ai/api/pricing (public): model IDs, groups, endpoint types, tags, price ratios.
  * LiteLLM's model_prices_and_context_window.json: context size, output limit and capability
    flags of the same upstream model (ModAPI's public catalog does not publish limits).
Price conversion (verified 2026-11-20 on 64 models against vendor list prices):
  input $/1M = model_ratio * 2; output = input * completion_ratio. LLM_Standard group (ratio 1).
Only models that are in LLM_Standard, speak the OpenAI endpoint, bill per token, and have a
LiteLLM row with limits are shipped. Dated snapshots and deprecated models are left out; users
can still add any model ID by hand (customizable-model).
"""
import json
import re
import sys
from pathlib import Path

import yaml

PLUGIN = Path(sys.argv[1])
PRICING = json.loads(Path(sys.argv[2]).read_text())["data"]
COST = json.loads(Path(sys.argv[3]).read_text())

DATED = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{4}|\d{6})$|-preview$|-latest$|-\d{4}-preview$")
# Vendor-stated specs checked by the ModAPI team on 2026-11-20; not channel-tested guarantees.
# context, max output (None = unconfirmed, leave the max_tokens limit open), output parameter name, image input allowed
SWAIN = {
    "kimi-k3": (1048576, 1048576, "max_completion_tokens", True),
    "MiniMax-M3.1-Flash": (1000000, 524288, "max_completion_tokens", True),
    "deepseek-v4.1-flash": (1000000, 393216, "max_tokens", True),
    "glm-5.3": (1048576, 131072, "max_tokens", True),
    "qwen3.8-max": (1000000, 131072, "max_tokens", True),
    "gpt-oss-120b": (131072, 131072, "max_tokens", True),
    # Vendor pages give only K/M values for these five. Rounded down conservatively (K = 1,000, M = 1,000,000,
    # never the binary 1,024 multiples) so Dify never allows more than the vendor states. Noted in the PR.
    "glm-5.3-flash": (1000000, 128000, "max_tokens", True),
    "glm-4.5": (128000, 96000, "max_tokens", True),
    "glm-4-airx": (8000, 4000, "max_tokens", True),
    "glm-4-flash": (128000, 4000, "max_tokens", True),
    "glm-4-long": (1000000, 4000, "max_tokens", True),
}
# Max output not confirmed by any source (checked 2026-11-20). Not preset; users can still add them by ID.
HOLD = {"grok-4", "deepseek-v3-1", "qwq-32b"}
VENDOR_ORDER = ["gpt-", "o", "claude-", "gemini-", "grok-", "deepseek-", "qwen", "kimi-", "glm-", "minimax"]


def vendor_row(name: str):
    for key in (name, f"openai/{name}", f"anthropic/{name}", f"gemini/{name}", f"xai/{name}",
                f"deepseek/{name}", f"vertex_ai/{name}"):
        row = COST.get(key)
        if row and row.get("max_input_tokens") and row.get("input_cost_per_token") is not None:
            return row
    return None


def money(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".") or "0"


def is_reasoning_only(name: str, row: dict) -> bool:
    # OpenAI reasoning models reject temperature/top_p.
    return bool(row.get("supports_reasoning")) and (name.startswith(("o1", "o3", "o4")) or name.startswith("gpt-5"))


def swain_yaml(item: dict) -> dict:
    name = item["model_name"]
    context, max_out, out_param, image_ok = SWAIN[name]
    tags = set((item.get("tags") or "").split(","))
    features = []
    if "tools" in tags:
        features += ["tool-call", "multi-tool-call", "stream-tool-call", "agent-thought"]
    if image_ok and tags & {"vision", "multimodal"}:
        features.append("vision")
    out_rule = {"name": out_param, "use_template": "max_tokens", "required": False, "min": 1}
    if max_out:
        out_rule.update(default=min(4096, max_out), max=max_out)
    else:
        out_rule["default"] = 4096
    inp = item["model_ratio"] * 2
    return {
        "model": name,
        "label": {"en_US": name},
        "model_type": "llm",
        "features": features,
        "model_properties": {"mode": "chat", "context_size": context},
        "parameter_rules": [{"name": "temperature", "use_template": "temperature"},
                            {"name": "top_p", "use_template": "top_p"}, out_rule],
        "pricing": {"input": money(inp), "output": money(inp * item.get("completion_ratio", 1)),
                    "unit": "0.000001", "currency": "USD"},
    }


def llm_yaml(item: dict, row: dict) -> dict:
    name = item["model_name"]
    tags = set((item.get("tags") or "").split(","))
    tools = "tools" in tags or bool(row.get("supports_function_calling"))
    vision = "vision" in tags or bool(row.get("supports_vision"))
    features = []
    if tools:
        features += ["tool-call", "multi-tool-call", "stream-tool-call", "agent-thought"]
    if vision:
        features.append("vision")
    if row.get("supports_response_schema"):
        features.append("structured-output")
    max_out = int(row.get("max_output_tokens") or row.get("max_tokens") or 4096)
    max_out = min(max_out, int(row["max_input_tokens"]))
    rules = []
    if not is_reasoning_only(name, row):
        rules += [{"name": "temperature", "use_template": "temperature"},
                  {"name": "top_p", "use_template": "top_p"}]
    rules.append({"name": "max_tokens", "use_template": "max_tokens",
                  "default": min(4096, max_out), "min": 1, "max": max_out})
    if row.get("supports_reasoning") and name.startswith(("gpt-5", "o")):
        rules.append({"name": "reasoning_effort", "label": {"en_US": "Reasoning effort"}, "type": "string",
                      "required": False, "options": ["low", "medium", "high"],
                      "help": {"en_US": "How much the model reasons before it answers. Lower is faster and cheaper."}})
    rules += [{"name": "response_format", "label": {"en_US": "Response format"}, "type": "string",
               "required": False, "options": ["text", "json_object"] + (["json_schema"] if row.get("supports_response_schema") else []),
               "help": {"en_US": "Plain text, any JSON object, or JSON that follows a schema."}}]
    if row.get("supports_response_schema"):
        rules.append({"name": "json_schema", "use_template": "json_schema"})
    inp = item["model_ratio"] * 2
    return {
        "model": name,
        "label": {"en_US": name},
        "model_type": "llm",
        "features": features,
        "model_properties": {"mode": "chat", "context_size": int(row["max_input_tokens"])},
        "parameter_rules": rules,
        "pricing": {"input": money(inp), "output": money(inp * item.get("completion_ratio", 1)),
                    "unit": "0.000001", "currency": "USD"},
    }


def embedding_yaml(item: dict, row: dict) -> dict:
    inp = item["model_ratio"] * 2
    return {
        "model": item["model_name"],
        "label": {"en_US": item["model_name"]},
        "model_type": "text-embedding",
        "model_properties": {"context_size": int(row["max_input_tokens"]), "max_chunks": 32},
        "pricing": {"input": money(inp), "unit": "0.000001", "currency": "USD"},
    }


def order_key(name: str):
    for i, p in enumerate(VENDOR_ORDER):
        if name.startswith(p):
            return (i, name)
    return (len(VENDOR_ORDER), name)


def main() -> None:
    llm, emb, skipped = {}, {}, []
    for item in PRICING:
        name = item["model_name"]
        tags = set((item.get("tags") or "").split(","))
        ok = ("LLM_Standard" in (item.get("enable_groups") or []) and "openai" in (item.get("supported_endpoint_types") or [])
              and item.get("quota_type") == 0 and item.get("model_ratio") is not None)
        if not ok:
            continue
        if name in HOLD:
            skipped.append((name, "max output unconfirmed, add by ID"))
            continue
        if name in SWAIN:
            llm[name] = swain_yaml(item)
            continue
        if "deprecated" in tags or DATED.search(name):
            skipped.append((name, "dated/deprecated"))
            continue
        row = vendor_row(name)
        if row is None:
            skipped.append((name, "no context data"))
            continue
        if "text-embeddings" in tags or "embedding" in (item.get("supported_endpoint_types") or []):
            emb[name] = embedding_yaml(item, row)
        elif row.get("mode", "chat") == "chat":
            llm[name] = llm_yaml(item, row)
    for kind, models in (("llm", llm), ("text_embedding", emb)):
        d = PLUGIN / "models" / kind
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.yaml"):
            if old.name != "_position.yaml":
                old.unlink()
        names = sorted(models, key=order_key)
        for n in names:
            (d / (re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-") + ".yaml")).write_text(
                yaml.safe_dump(models[n], sort_keys=False, allow_unicode=False, width=100))
        (d / "_position.yaml").write_text(yaml.safe_dump(names, sort_keys=False))
    print(f"llm {len(llm)} | text_embedding {len(emb)} | skipped {len(skipped)}")
    for n, why in skipped:
        print(f"  skip {n}: {why}")


if __name__ == "__main__":
    main()
