# Privacy Policy

This policy covers the ModAPI model provider plugin for Dify, published by Modlane Systems LLC.

## What the plugin sends

When you use a ModAPI model in Dify, the plugin sends the request to the ModAPI API at `https://modapi.ai/v1`. A request carries:

- the messages of the conversation, including any images you attach, or the texts to embed;
- tool definitions and tool results, when the app uses tools;
- the model settings you chose, such as the output token limit;
- your ModAPI API key, to authenticate the request.

When you save credentials or add a model, the plugin also calls `GET https://modapi.ai/v1/models` to check the key, and reads the public model catalog at `https://modapi.ai/api/pricing`. These requests carry no conversation content; the catalog request carries no key.

## What the plugin stores and logs

The plugin does not collect, store or log your messages, files or credentials. It keeps nothing between requests except a copy of the public model catalog in memory for up to ten minutes. Dify stores the credentials you enter in its own encrypted storage.

## How ModAPI handles requests

For the stateless text-inference requests covered by its data retention statement, ModAPI does not write prompt or response content to databases, logs or backups, and keeps only billing and usage metadata. ModAPI forwards each request to the company that provides the model you picked, such as OpenAI, Anthropic, Google or xAI, whose own policy applies on standard routes. Details are in the [API Data Retention and ZDR Statement](https://modapi.ai/policies/zdr/) and the [ModAPI Privacy Policy](https://modapi.ai/privacy-policy).

## Contact

Questions about this policy: support@modapi.ai.
