# ModAPI

**Author:** modapi · **Version:** 0.1.0 · **Type:** model provider
**Source:** https://github.com/modapi-ai/dify-plugin · **Contact:** support@modapi.ai

Use chat and embedding models from OpenAI, Anthropic, Google, xAI and other vendors in Dify with one [ModAPI](https://modapi.ai) API key.

## What this plugin adds

Without this plugin, a ModAPI user goes through the OpenAI-API-compatible provider and types in every model by hand, with its context size, output limit, vision and tool support. This plugin ships those values filled in:

- **Predefined models** with context size, output limit, list price per million tokens, and tool calling, vision and structured output flags.
- **Any other model by ID.** Add a model from the [ModAPI catalog](https://modapi.ai/pricing) by its exact ID; the plugin checks that the ID exists and is a chat or embedding model before saving it, and fills in its price.
- **Free key check.** Saving the key calls `GET /v1/models`, which costs nothing, instead of a paid chat request.
- **Real token counts on streams.** Streaming requests ask for usage in the last chunk, so Dify records the tokens ModAPI bills.

Controls that live on the ModAPI key apply to every request from Dify: each key can have its own model allowlist, IP allowlist, spending quota, expiry date, channel group, and cross-group retry and failover order. For the text requests covered by the [data retention statement](https://modapi.ai/policies/zdr/), the LLM_Prime channel group uses upstream zero-data-retention routes by default.

## Setup

1. Create an account at [modapi.ai](https://modapi.ai) and create an API key at [modapi.ai/keys](https://modapi.ai/keys). Set the key's model allowlist, quota and channel group there if you need them.
2. In Dify, open **Settings → Model Provider**, find **ModAPI**, and paste the key.
3. Pick a ModAPI model in any LLM or Knowledge node.

To add a model that is not in the list, choose **Add Model**, enter the model ID exactly as it appears on [modapi.ai/pricing](https://modapi.ai/pricing), and optionally its context size and output limit.

## Requirements

- A ModAPI API key.
- Network access from the Dify plugin runtime to `https://modapi.ai`.

## Models and prices

GPT-5 series models have a 400K total context window, of which at most 272K can be input; the plugin sets their Dify context size to the 272K input limit so requests never exceed it.

Where a vendor states a limit only as "128K" or "1M", the plugin rounds it down to a decimal integer (128,000 or 1,000,000) rather than the binary 131,072 or 1,048,576, so Dify never allows more than the vendor states. Models whose maximum output is not published, such as `grok-4`, `deepseek-v3-1` and `qwq-32b`, are not predefined; add them by ID if you need them.

Listed prices are the standard channel group prices at the time of release, in USD per million tokens. ModAPI bills the actual usage at the price of the channel group your key uses; check [modapi.ai/pricing](https://modapi.ai/pricing) for current prices.

Image, video and audio generation models on ModAPI are not Dify model types and are not part of this plugin.

## Privacy

See [PRIVACY.md](PRIVACY.md).
