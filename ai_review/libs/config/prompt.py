from functools import cached_property
from pathlib import Path
from typing import Union

import requests

from pydantic import BaseModel, Field, AnyUrl, FilePath

from ai_review.libs.resources import load_resource


def resolve_prompt_files(files: list[Union[FilePath, AnyUrl]] | None, default_file: str) -> list[Union[Path, AnyUrl]]:
    return files or [
        load_resource(
            package="ai_review.prompts",
            filename=default_file,
            fallback=f"ai_review/prompts/{default_file}"
        )
    ]


def resolve_system_prompt_files(files: list[Union[FilePath, AnyUrl]] | None, include: bool, default_file: str) -> list[Union[Path, AnyUrl]]:
    global_files = [
        load_resource(
            package="ai_review.prompts",
            filename=default_file,
            fallback=f"ai_review/prompts/{default_file}"
        )
    ]

    if files is None:
        return global_files

    if include:
        return global_files + files

    return files


class PromptConfig(BaseModel):
    context: dict[str, str] = Field(default_factory=dict)
    normalize_prompts: bool = True
    context_placeholder: str = "<<{value}>>"

    # --- Prompts ---
    inline_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    context_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    summary_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    inline_reply_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    summary_reply_prompt_files: list[Union[FilePath, AnyUrl]] | None = None

    # --- System Prompts ---
    system_inline_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    system_context_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    system_summary_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    system_inline_reply_prompt_files: list[Union[FilePath, AnyUrl]] | None = None
    system_summary_reply_prompt_files: list[Union[FilePath, AnyUrl]] | None = None

    # --- Include System Prompts ---
    include_inline_system_prompts: bool = True
    include_context_system_prompts: bool = True
    include_summary_system_prompts: bool = True
    include_inline_reply_system_prompts: bool = True
    include_summary_reply_system_prompts: bool = True

    # --- Prompts ---
    @cached_property
    def inline_prompt_files_or_default(self) -> list[Path]:
        return resolve_prompt_files(self.inline_prompt_files, "default_inline.md")

    @cached_property
    def context_prompt_files_or_default(self) -> list[Path]:
        return resolve_prompt_files(self.context_prompt_files, "default_context.md")

    @cached_property
    def summary_prompt_files_or_default(self) -> list[Path]:
        return resolve_prompt_files(self.summary_prompt_files, "default_summary.md")

    @cached_property
    def inline_reply_prompt_files_or_default(self) -> list[Path]:
        return resolve_prompt_files(self.inline_reply_prompt_files, "default_inline_reply.md")

    @cached_property
    def summary_reply_prompt_files_or_default(self) -> list[Path]:
        return resolve_prompt_files(self.summary_reply_prompt_files, "default_summary_reply.md")

    # --- System Prompts ---
    @cached_property
    def system_inline_prompt_files_or_default(self) -> list[Path]:
        return resolve_system_prompt_files(
            files=self.system_inline_prompt_files,
            include=self.include_inline_system_prompts,
            default_file="default_system_inline.md"
        )

    @cached_property
    def system_context_prompt_files_or_default(self) -> list[Path]:
        return resolve_system_prompt_files(
            files=self.system_context_prompt_files,
            include=self.include_context_system_prompts,
            default_file="default_system_context.md"
        )

    @cached_property
    def system_summary_prompt_files_or_default(self) -> list[Path]:
        return resolve_system_prompt_files(
            files=self.system_summary_prompt_files,
            include=self.include_summary_system_prompts,
            default_file="default_system_summary.md"
        )

    @cached_property
    def system_inline_reply_prompt_files_or_default(self) -> list[Path]:
        return resolve_system_prompt_files(
            files=self.system_inline_reply_prompt_files,
            include=self.include_inline_reply_system_prompts,
            default_file="default_system_inline_reply.md"
        )

    @cached_property
    def system_summary_reply_prompt_files_or_default(self) -> list[Path]:
        return resolve_system_prompt_files(
            files=self.system_summary_reply_prompt_files,
            include=self.include_summary_reply_system_prompts,
            default_file="default_system_summary_reply.md"
        )

    def _load_prompt_content(self, source: Union[Path, AnyUrl]) -> str:
        """Load prompt content from a local file or URL."""
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")
        elif source.scheme in ("http", "https"):
            response = requests.get(str(source))
            response.raise_for_status()  # Raise an error for bad status codes
            return response.text
        else:
            raise ValueError(f"Unsupported prompt source type: {type(source)} or scheme: {source.scheme}")

    # --- Load Prompts ---
    def load_inline(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.inline_prompt_files_or_default]

    def load_context(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.context_prompt_files_or_default]

    def load_summary(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.summary_prompt_files_or_default]

    def load_inline_reply(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.inline_reply_prompt_files_or_default]

    def load_summary_reply(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.summary_reply_prompt_files_or_default]

    # --- Load System Prompts ---
    def load_system_inline(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.system_inline_prompt_files_or_default]

    def load_system_context(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.system_context_prompt_files_or_default]

    def load_system_summary(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.system_summary_prompt_files_or_default]

    def load_system_inline_reply(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.system_inline_reply_prompt_files_or_default]

    def load_system_summary_reply(self) -> list[str]:
        return [self._load_prompt_content(source) for source in self.system_summary_reply_prompt_files_or_default]
