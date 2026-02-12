import pytest
from ai_review.config import settings
from ai_review.libs.llm.tokenizer import count_tokens
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.prompt.service import PromptService


def test_large_prompt_token_counting():
    """Test token counting for very large prompts."""
    # Create a very large prompt
    large_text = "x" * 100000  # ~25,000 tokens
    tokens = count_tokens(large_text)
    assert isinstance(tokens, int)
    assert tokens > 0


def test_prompt_chunking_small_prompt(fake_prompt_context: PromptContextSchema):
    """Test that small prompts are not chunked."""
    diffs = [
        DiffFileSchema(file="a.py", diff="+ small change"),
    ]
    prompt = PromptService.build_summary_request(diffs, fake_prompt_context)
    chunks = PromptService.split_prompt(prompt, max_tokens=2000)

    assert len(chunks) == 1
    assert prompt in chunks[0]


def test_prompt_chunking_large_prompt():
    """Test that large prompts are properly chunked."""
    # Create a very large prompt
    large_content = "x" * 10000  # ~2,500 tokens
    large_prompt = f"## Changes\n\n# File: large_file.py\n{large_content}"

    # Split into chunks of 1000 tokens (~4000 characters)
    chunks = PromptService.split_prompt(large_prompt, max_tokens=1000)

    assert len(chunks) > 1
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


def test_prompt_chunking_with_system_prompt(fake_prompt_context: PromptContextSchema):
    """Test chunking with system prompt."""
    diffs = [
        DiffFileSchema(file="a.py", diff="+ change 1"),
        DiffFileSchema(file="b.py", diff="- change 2"),
        DiffFileSchema(file="c.py", diff="* change 3"),
    ]
    user_prompt = PromptService.build_summary_request(diffs, fake_prompt_context)
    system_prompt = PromptService.build_system_summary_request(fake_prompt_context)

    # Split with system prompt
    chunks = PromptService.split_prompt(user_prompt, max_tokens=500, system_prompt=system_prompt)

    assert len(chunks) >= 1
    assert all(isinstance(chunk, str) and len(chunk) > 0 for chunk in chunks)


def test_prompt_chunking_file_level_splitting():
    """Test that prompt is split at file boundaries when possible."""
    # Create a prompt with multiple files with very large content
    large_content = "x" * 2000  # ~500 tokens per file
    prompt = f"""## Changes

# File: file1.py
{large_content}

# File: file2.py
{large_content}

# File: file3.py
{large_content}
"""

    # Split into chunks
    chunks = PromptService.split_prompt(prompt, max_tokens=600)

    # Should split into 4 sections: 1 header section + 3 file sections
    assert len(chunks) == 4
    assert chunks[0].strip() == "## Changes"
    # Check each file is in its own chunk
    file_chunks = chunks[1:]
    assert len(file_chunks) == 3
    assert all("File:" in chunk for chunk in file_chunks)
    assert all(("file1.py" in file_chunks[0], "file2.py" in file_chunks[1], "file3.py" in file_chunks[2]))


def test_prompt_chunking_single_large_file():
    """Test handling of a single very large file."""
    large_content = "x" * 8000  # ~2000 tokens
    prompt = f"## Changes\n\n# File: huge_file.py\n{large_content}"

    chunks = PromptService.split_prompt(prompt, max_tokens=1000)

    # Should split the large file into multiple chunks
    assert len(chunks) > 1
    
    # Check that file header is in each file chunk
    file_chunks = [c for c in chunks if c.strip() and "File:" in c]
    assert len(file_chunks) > 1
    assert all("huge_file.py" in chunk for chunk in file_chunks)
    
    # Check that all x's are included in the chunks
    all_content = "".join(chunks)
    assert all_content.count("x") == 8000
