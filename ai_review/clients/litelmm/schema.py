from typing import Literal

from pydantic import BaseModel


class LiteLLMUsageSchema(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class LiteLLMMessageSchema(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LiteLLMChoiceSchema(BaseModel):
    message: LiteLLMMessageSchema


class LiteLLMChatRequestSchema(BaseModel):
    model: str
    messages: list[LiteLLMMessageSchema]
    max_tokens: int | None = None
    temperature: float | None = None


class LiteLLMChatResponseSchema(BaseModel):
    usage: LiteLLMUsageSchema
    choices: list[LiteLLMChoiceSchema]

    @property
    def first_text(self) -> str:
        if not self.choices:
            return ""

        return (self.choices[0].message.content or "").strip()
