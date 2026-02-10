from typing import Any

import pytest
from pydantic import HttpUrl, SecretStr

from ai_review.clients.litelmm.schema import (
    LiteLLMUsageSchema,
    LiteLLMChoiceSchema,
    LiteLLMMessageSchema,
    LiteLLMChatRequestSchema,
    LiteLLMChatResponseSchema,
)
from ai_review.clients.litelmm.types import LiteLLMHTTPClientProtocol
from ai_review.config import settings
from ai_review.libs.config.llm.base import LiteLLMLLMConfig
from ai_review.libs.config.llm.litelmm import LiteLLMMetaConfig, LiteLLMHTTPClientConfig
from ai_review.libs.constants.llm_provider import LLMProvider
from ai_review.services.llm.litelmm.client import LiteLLMLLMClient


class FakeLiteLLMHTTPClient(LiteLLMHTTPClientProtocol):
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or {}

    async def chat(self, request: LiteLLMChatRequestSchema) -> LiteLLMChatResponseSchema:
        self.calls.append(("chat", {"request": request}))
        return self.responses.get(
            "chat",
            LiteLLMChatResponseSchema(
                usage=LiteLLMUsageSchema(total_tokens=15, prompt_tokens=8, completion_tokens=7),
                choices=[
                    LiteLLMChoiceSchema(
                        message=LiteLLMMessageSchema(
                            role="assistant",
                            content="FAKE_LITELLM_RESPONSE"
                        )
                    )
                ],
            ),
        )


@pytest.fixture
def fake_litelmm_http_client() -> FakeLiteLLMHTTPClient:
    return FakeLiteLLMHTTPClient()


@pytest.fixture
def litelmm_llm_client(
        monkeypatch: pytest.MonkeyPatch,
        fake_litelmm_http_client: FakeLiteLLMHTTPClient,
) -> LiteLLMLLMClient:
    monkeypatch.setattr(
        "ai_review.services.llm.litelmm.client.get_litelmm_http_client",
        lambda: fake_litelmm_http_client,
    )
    return LiteLLMLLMClient()


@pytest.fixture
def litelmm_http_client_config(monkeypatch: pytest.MonkeyPatch):
    fake_config = LiteLLMLLMConfig(
        meta=LiteLLMMetaConfig(
            model="gpt-4o-mini",
            max_tokens=1200,
            temperature=0.3
        ),
        provider=LLMProvider.LITELLM,
        http_client=LiteLLMHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.litellm.ai"),
            api_token=SecretStr("fake-token"),
        ),
    )
    monkeypatch.setattr(settings, "llm", fake_config)
