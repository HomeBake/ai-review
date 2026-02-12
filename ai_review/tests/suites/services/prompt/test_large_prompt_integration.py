#!/usr/bin/env python3
"""
Test script to demonstrate large prompt handling functionality.

This script creates a large prompt and tests the tokenizer and prompt chunking
functionality.
"""

from ai_review.libs.llm.tokenizer import count_tokens, TokenizerFactory
from ai_review.services.prompt.service import PromptService


def test_tokenizer():
    """Test tokenizer functionality."""
    print("=== Tokenizer Test ===")
    
    # Test with different model types
    test_text = "This is a test for tokenizer functionality with large prompts. " * 100
    print(f"Test text length: {len(test_text)} characters")
    
    # GPT tokenizer
    gpt_tokens = count_tokens(test_text, "gpt-4")
    print(f"GPT-4 token count: {gpt_tokens}")
    
    # Claude tokenizer
    claude_tokens = count_tokens(test_text, "claude-3")
    print(f"Claude-3 token count: {claude_tokens}")
    
    # Llama tokenizer (LiteLLM)
    llama_tokens = count_tokens(test_text, "llama-3")
    print(f"Llama-3 token count: {llama_tokens}")
    
    # Default tokenizer
    default_tokens = count_tokens(test_text)
    print(f"Default token count: {default_tokens}")
    
    return test_text


def test_prompt_chunking(text):
    """Test prompt chunking functionality."""
    print("\n=== Prompt Chunking Test ===")
    
    # Create a large prompt
    large_prompt = (
        "## Changes\n\n"
        "# File: large_file.py\n" + text + "\n\n"
        "# File: another_large_file.py\n" + text + "\n\n"
        "# File: very_large_file.py\n" + text
    )
    
    total_tokens = count_tokens(large_prompt)
    print(f"Total prompt token count: {total_tokens}")
    
    # Test chunking with different max token limits
    for max_tokens in [1000, 2000, 3000]:
        chunks = PromptService.split_prompt(large_prompt, max_tokens=max_tokens)
        print(f"\nChunking with max {max_tokens} tokens:")
        print(f"  Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            print(f"  Chunk {i+1}: {chunk_tokens} tokens")
            
            # Check token count is within limit
            assert chunk_tokens <= max_tokens, f"Chunk {i+1} exceeds limit"


def test_file_level_chunking(text):
    """Test file-level chunking."""
    print("\n=== File Level Chunking Test ===")
    
    large_prompt = (
        "## Changes\n\n"
        "# File: file1.py\n" + text + "\n\n"
        "# File: file2.py\n" + text + "\n\n"
        "# File: file3.py\n" + text
    )
    
    # Split with small token limit to force each file into its own chunk
    chunks = PromptService.split_prompt(large_prompt, max_tokens=1500)
    
    print(f"Number of chunks: {len(chunks)}")
    
    # Verify each chunk contains a file
    file_chunks = [c for c in chunks if "# File:" in c]
    print(f"Number of file chunks: {len(file_chunks)}")
    
    assert len(file_chunks) == 3, "Should have 3 file chunks"
    
    # Check each file is present
    for i, file_chunk in enumerate(file_chunks):
        expected_file = f"file{i+1}.py"
        assert expected_file in file_chunk, f"Chunk should contain {expected_file}"
        print(f"  File {i+1} chunk: {len(file_chunk)} characters")


if __name__ == "__main__":
    try:
        print("Testing Large Prompt Handling in AI Review Tool")
        print("=" * 60)
        
        # Test tokenizer
        test_text = test_tokenizer()
        
        # Test prompt chunking
        test_prompt_chunking(test_text)
        
        # Test file level chunking
        test_file_level_chunking(test_text)
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Large prompt handling is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(f"Stack trace:\n{traceback.format_exc()}")
