from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.libs.llm.tokenizer import count_tokens
from ai_review.services.diff.schema import DiffFileSchema
from ai_review.services.prompt.schema import PromptContextSchema
from ai_review.services.prompt.tools import normalize_prompt, format_file, format_thread, format_files
from ai_review.services.prompt.types import PromptServiceProtocol
from ai_review.services.vcs.types import ReviewThreadSchema

logger = get_logger("PROMPT_SERVICE")


class PromptService(PromptServiceProtocol):
    @classmethod
    def prepare_prompt(cls, prompts: list[str], context: PromptContextSchema) -> str:
        prompt = "\n\n".join(prompts)
        prompt = context.apply_format(prompt)

        if settings.prompt.normalize_prompts:
            prompt = normalize_prompt(prompt)

        return prompt

    @classmethod
    def build_inline_request(cls, diff: DiffFileSchema, context: PromptContextSchema) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_inline(), context)
        return (
            f"{prompt}\n\n"
            f"## Diff\n\n"
            f"{format_file(diff)}"
        )

    @classmethod
    def build_summary_request(cls, diffs: list[DiffFileSchema], context: PromptContextSchema) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_summary(), context)
        changes = format_files(diffs)
        return (
            f"{prompt}\n\n"
            f"## Changes\n\n"
            f"{changes}\n"
        )

    @classmethod
    def build_context_request(cls, diffs: list[DiffFileSchema], context: PromptContextSchema) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_context(), context)
        changes = format_files(diffs)
        return (
            f"{prompt}\n\n"
            f"## Diff\n\n"
            f"{changes}\n"
        )

    @classmethod
    def build_inline_reply_request(
            cls,
            diff: DiffFileSchema,
            thread: ReviewThreadSchema,
            context: PromptContextSchema
    ) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_inline_reply(), context)
        conversation = format_thread(thread)

        return (
            f"{prompt}\n\n"
            f"## Conversation\n\n"
            f"{conversation}\n\n"
            f"## Diff\n\n"
            f"{format_file(diff)}"
        )

    @classmethod
    def build_summary_reply_request(
            cls,
            diffs: list[DiffFileSchema],
            thread: ReviewThreadSchema,
            context: PromptContextSchema
    ) -> str:
        prompt = cls.prepare_prompt(settings.prompt.load_summary_reply(), context)
        changes = format_files(diffs)
        conversation = format_thread(thread)

        return (
            f"{prompt}\n\n"
            f"## Conversation\n\n"
            f"{conversation}\n\n"
            f"## Changes\n\n"
            f"{changes}"
        )

    @classmethod
    def build_system_inline_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline(), context)

    @classmethod
    def build_system_context_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_context(), context)

    @classmethod
    def build_system_summary_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary(), context)

    @classmethod
    def build_system_inline_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_inline_reply(), context)

    @classmethod
    def build_system_summary_reply_request(cls, context: PromptContextSchema) -> str:
        return cls.prepare_prompt(settings.prompt.load_system_summary_reply(), context)

    @classmethod
    def split_prompt(cls, prompt: str, max_tokens: int, system_prompt: str = "") -> list[str]:
        """
        Split a large prompt into chunks that fit within the maximum token limit.

        Args:
            prompt: The prompt to split
            max_tokens: Maximum token limit per chunk
            system_prompt: System prompt (will be included in each chunk)

        Returns:
            List of prompt chunks
        """
        # Calculate system prompt token count
        system_tokens = count_tokens(system_prompt) if system_prompt else 0
        # Calculate available tokens for user prompt content
        available_tokens = max_tokens - system_tokens - 100  # Reserve some tokens for completion

        if available_tokens <= 0:
            logger.warning("System prompt exceeds max token limit, returning empty chunks")
            return []

        # If prompt fits, return as single chunk
        if count_tokens(prompt) <= available_tokens:
            return [prompt]

        # Split by sections
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Try to split by file sections first (for summary prompts with multiple files)
        lines = prompt.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # If we see a file header, try to extract the entire file section
            if line.startswith("# File:"):
                # Add previous chunk if it has content
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Collect entire file section
                file_section = [line]
                i += 1

                while i < len(lines) and not lines[i].startswith("# File:"):
                    file_section.append(lines[i])
                    i += 1

                file_text = "\n".join(file_section)
                file_tokens = count_tokens(file_text)

                if file_tokens <= available_tokens:
                    chunks.append(file_text)
                else:
                    # If single file is too large, split into smaller pieces
                    logger.warning(f"File section too large ({file_tokens} tokens), splitting further")
                    sub_chunks = cls._split_large_section(file_text, available_tokens)
                    chunks.extend(sub_chunks)
            else:
                # For other lines, add incrementally
                line_tokens = count_tokens(line)

                if current_tokens + line_tokens <= available_tokens:
                    current_chunk.append(line)
                    current_tokens += line_tokens
                    i += 1
                else:
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    else:
                        # Single line is too long, split it
                        logger.warning(f"Line too long ({line_tokens} tokens), splitting")
                        sub_chunks = cls._split_large_line(line, available_tokens)
                        chunks.extend(sub_chunks)
                        i += 1

        # Add remaining content
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    @classmethod
    def _split_large_section(cls, section: str, max_tokens: int) -> list[str]:
        """Split a large section into smaller chunks."""
        lines = section.split("\n")
        chunks = []
        current_chunk = []
        current_tokens = 0
        header = lines[0] if lines and lines[0].startswith("# File:") else None

        for i, line in enumerate(lines):
            line_tokens = count_tokens(line)

            if current_tokens + line_tokens <= max_tokens:
                current_chunk.append(line)
                current_tokens += line_tokens
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                    # If we have a header, start new chunk with header
                    if header and i > 0:
                        current_chunk.append(header)
                        current_tokens = count_tokens(header)

                # Try to add the current line again
                if count_tokens(line) <= max_tokens:
                    if not current_chunk and header:
                        current_chunk.append(header)
                        current_tokens = count_tokens(header)
                    current_chunk.append(line)
                    current_tokens += line_tokens
                else:
                    # Single line is too long
                    sub_chunks = cls._split_large_line(line, max_tokens)
                    # If we have a header, prepend it to each sub-chunk
                    if header:
                        sub_chunks = [f"{header}\n{chunk}" for chunk in sub_chunks]
                    chunks.extend(sub_chunks)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    @classmethod
    def _split_large_line(cls, line: str, max_tokens: int) -> list[str]:
        """Split a single very long line into smaller chunks."""
        avg_chars_per_token = 4
        max_chars = max_tokens * avg_chars_per_token
        chunks = []

        for i in range(0, len(line), max_chars):
            chunk = line[i:i + max_chars]
            chunks.append(chunk)

        return chunks
