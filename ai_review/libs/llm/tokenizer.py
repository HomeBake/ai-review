"""
Tokenization library for AI Review tool.

This module provides tokenization utilities to measure prompt sizes for different LLM models.
The tokenizer supports multiple LLM providers and falls back to a default implementation
if a specific tokenizer is not available.
"""

import sys
from typing import Protocol, Optional
from ai_review.libs.logger import get_logger

logger = get_logger("TOKENIZER")


class TokenizerProtocol(Protocol):
    """Interface for tokenizer implementations."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...


class TiktokenTokenizer:
    """
    Tokenizer using OpenAI's tiktoken library.

    This tokenizer supports multiple encoding types and can be configured for different
    LLM models. If tiktoken is not available, it falls back to a simple character count
    approximation.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        self.encoding = encoding
        self._tokenizer = None
        self._use_fallback = False

        # Try to import tiktoken
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(encoding)
        except ImportError:
            logger.warning("tiktoken library not available, using fallback tokenizer")
            self._use_fallback = True
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken with encoding {encoding}: {e}, using fallback")
            self._use_fallback = True

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        If tiktoken is available, it uses the specified encoding. Otherwise, it falls
        back to a simple character count approximation (assuming ~4 characters per token).
        """
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenization failed: {e}, using fallback")
                self._use_fallback = True

        # Fallback token counting: approximate by characters (rough estimate)
        return len(text) // 4 + 1


class SimpleTokenizer:
    """
    Simple fallback tokenizer.

    This tokenizer approximates token count by assuming an average token length.
    It's used when no specific tokenizer is available.
    """

    def __init__(self, average_chars_per_token: int = 4):
        self.average_chars_per_token = average_chars_per_token

    def count_tokens(self, text: str) -> int:
        """Approximate token count using average characters per token."""
        return len(text) // self.average_chars_per_token + 1


class TokenizerFactory:
    """Factory class to create tokenizer instances based on model type or provider."""

    @staticmethod
    def get_tokenizer(model: str = "default") -> TokenizerProtocol:
        """
        Get a tokenizer instance based on model type.

        Args:
            model: Model name or type to determine the tokenizer

        Returns:
            TokenizerProtocol instance
        """
        model = model.lower()

        # For OpenAI models (gpt-3.5-turbo, gpt-4, etc.) and Azure OpenAI
        if any(keyword in model for keyword in ["gpt", "openai", "azure"]):
            return TiktokenTokenizer("cl100k_base")

        # For Claude models
        if any(keyword in model for keyword in ["claude", "anthropic"]):
            return TiktokenTokenizer("cl100k_base")  # Claude uses similar tokenization

        # For Gemini models
        if "gemini" in model:
            return TiktokenTokenizer("cl100k_base")  # Gemini uses similar tokenization

        # For Llama models (Ollama, LiteLLM)
        if any(keyword in model for keyword in ["llama", "ollama", "litelmm"]):
            return TiktokenTokenizer("cl100k_base")  # Llama 3 uses cl100k_base

        # Default tokenizer
        logger.debug(f"No specific tokenizer for model '{model}', using default")
        return TiktokenTokenizer("cl100k_base")


# Default global tokenizer for general use
_default_tokenizer = TokenizerFactory.get_tokenizer()


def count_tokens(text: str, model: str = "default") -> int:
    """
    Count the number of tokens in the given text.

    Args:
        text: Text to count tokens for
        model: Model name or type to determine tokenization method

    Returns:
        Number of tokens in the text
    """
    if model == "default":
        return _default_tokenizer.count_tokens(text)
    return TokenizerFactory.get_tokenizer(model).count_tokens(text)
