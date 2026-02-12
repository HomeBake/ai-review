import pytest
from ai_review.libs.llm.tokenizer import (
    count_tokens,
    TokenizerFactory,
    TiktokenTokenizer,
    SimpleTokenizer,
)


def test_count_tokens_default():
    """Test default token counting functionality."""
    text = "Hello, world!"
    tokens = count_tokens(text)
    assert isinstance(tokens, int)
    assert tokens > 0


def test_count_tokens_specific_model():
    """Test token counting with specific model name."""
    text = "Hello, world!"
    tokens_gpt = count_tokens(text, "gpt-4")
    tokens_llama = count_tokens(text, "llama-3")

    assert isinstance(tokens_gpt, int)
    assert isinstance(tokens_llama, int)
    assert tokens_gpt > 0
    assert tokens_llama > 0


def test_tokenizer_factory():
    """Test tokenizer factory creation."""
    # Test different model types
    gpt_tokenizer = TokenizerFactory.get_tokenizer("gpt-4o")
    assert isinstance(gpt_tokenizer, TiktokenTokenizer)

    llama_tokenizer = TokenizerFactory.get_tokenizer("llama-3-70b")
    assert isinstance(llama_tokenizer, TiktokenTokenizer)

    claude_tokenizer = TokenizerFactory.get_tokenizer("claude-3-5-sonnet-20250219")
    assert isinstance(claude_tokenizer, TiktokenTokenizer)

    gemini_tokenizer = TokenizerFactory.get_tokenizer("gemini-pro")
    assert isinstance(gemini_tokenizer, TiktokenTokenizer)


def test_simple_tokenizer():
    """Test the simple fallback tokenizer."""
    tokenizer = SimpleTokenizer(average_chars_per_token=4)

    # Test short text
    assert tokenizer.count_tokens("Hello") == 2  # 5 chars → ~2 tokens
    assert tokenizer.count_tokens("Hi") == 1  # 2 chars → ~1 token

    # Test longer text
    long_text = "x" * 100
    assert tokenizer.count_tokens(long_text) == 26  # 100 / 4 = 25 + 1


def test_tiktoken_tokenizer():
    """Test tiktoken based tokenizer."""
    tokenizer = TiktokenTokenizer(encoding="cl100k_base")

    # Test basic functionality
    text = "Hello, world!"
    tokens = tokenizer.count_tokens(text)
    assert isinstance(tokens, int)
    assert tokens > 0

    # Test with longer text
    long_text = "This is a longer text that should contain multiple tokens. " * 10
    tokens = tokenizer.count_tokens(long_text)
    assert tokens > 10


def test_tokenizer_with_large_text():
    """Test tokenizer with very large text (simulating large prompts)."""
    # Create a very large text
    large_text = "x" * 10000  # Approximately 2500 tokens
    tokens = count_tokens(large_text)

    assert isinstance(tokens, int)
    assert tokens > 0


def test_tokenizer_for_litelmm():
    """Test tokenizer specifically for LiteLLM."""
    # Test LiteLLM with various model types
    tokens_gpt = count_tokens("Hello, world!", "gpt-4o")
    tokens_llama = count_tokens("Hello, world!", "llama-3")
    tokens_claude = count_tokens("Hello, world!", "claude-3")

    assert tokens_gpt > 0
    assert tokens_llama > 0
    assert tokens_claude > 0
