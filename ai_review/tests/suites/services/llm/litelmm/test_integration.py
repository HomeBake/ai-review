#!/usr/bin/env python3
"""Simple test to verify LiteLLM integration"""

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_review.libs.constants.llm_provider import LLMProvider
from ai_review.services.llm.factory import get_llm_client
from ai_review.services.llm.litelmm.client import LiteLLMLLMClient
from ai_review.libs.config.llm.litelmm import LiteLLMMetaConfig, LiteLLMHTTPClientConfig
from ai_review.libs.config.llm.base import LiteLLMLLMConfig
from ai_review.config import settings
from pydantic import HttpUrl

print("Testing LiteLLM integration...")

try:
    # Verify LLMProvider has LITELLM enum
    assert hasattr(LLMProvider, "LITELLM"), "LLMProvider.LITELLM not defined"
    print("✓ LLMProvider.LITELLM exists")
    
    # Verify we can import LiteLLMLLMClient
    print("✓ LiteLLMLLMClient import successful")
    
    # Verify LiteLLM config classes
    meta_config = LiteLLMMetaConfig(model="gpt-4o-mini", max_tokens=1200, temperature=0.3)
    http_config = LiteLLMHTTPClientConfig(
        timeout=30,
        api_url=HttpUrl("https://api.litellm.ai"),
        api_token="test-token"
    )
    llm_config = LiteLLMLLMConfig(
        meta=meta_config,
        provider=LLMProvider.LITELLM,
        http_client=http_config
    )
    print("✓ LiteLLM config classes initialized successfully")
    
    print("\nLiteLLM integration test PASSED!")
    print("\nNext steps:")
    print("1. To use LiteLLM, set LLM__PROVIDER=LITELLM in your .env file")
    print("2. Set LLM__HTTP_CLIENT__API_TOKEN with your LiteLLM API key")
    print("3. (Optional) Set LLM__HTTP_CLIENT__API_URL if using a custom LiteLLM proxy")
    
except Exception as e:
    print(f"\n✗ Test failed: {type(e).__name__}: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)
