from ai_review.libs.config.http import HTTPClientWithTokenConfig
from ai_review.libs.config.llm.meta import LLMMetaConfig


class LiteLLMMetaConfig(LLMMetaConfig):
    model: str = "gpt-4o-mini"


class LiteLLMHTTPClientConfig(HTTPClientWithTokenConfig):
    pass
