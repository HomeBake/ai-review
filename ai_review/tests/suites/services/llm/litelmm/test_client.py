import pytest

from ai_review.services.llm.litelmm.client import LiteLLMLLMClient
from ai_review.services.llm.types import ChatResultSchema
from ai_review.tests.fixtures.clients.litelmm import FakeLiteLLMHTTPClient


@pytest.mark.asyncio
@pytest.mark.usefixtures("litelmm_http_client_config")
async def test_litelmm_llm_chat(
        litelmm_llm_client: LiteLLMLLMClient,
        fake_litelmm_http_client: FakeLiteLLMHTTPClient
):
    result = await litelmm_llm_client.chat("prompt", "prompt_system")

    assert isinstance(result, ChatResultSchema)
    assert result.text == "FAKE_LITELLM_RESPONSE"
    assert result.total_tokens == 15
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 7

    assert fake_litelmm_http_client.calls[0][0] == "chat"
