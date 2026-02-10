from typing import Protocol

from ai_review.clients.litelmm.schema import LiteLLMChatRequestSchema, LiteLLMChatResponseSchema


class LiteLLMHTTPClientProtocol(Protocol):
    async def chat(self, request: LiteLLMChatRequestSchema) -> LiteLLMChatResponseSchema:
        ...
