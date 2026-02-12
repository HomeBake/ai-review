from ai_review.config import settings
from ai_review.libs.logger import get_logger
from ai_review.libs.llm.tokenizer import count_tokens
from ai_review.services.artifacts.types import ArtifactsServiceProtocol
from ai_review.services.cost.types import CostServiceProtocol
from ai_review.services.hook import hook
from ai_review.services.llm.types import LLMClientProtocol
from ai_review.services.review.gateway.types import ReviewLLMGatewayProtocol

logger = get_logger("REVIEW_LLM_GATEWAY")


class ReviewLLMGateway(ReviewLLMGatewayProtocol):
    def __init__(
            self,
            llm: LLMClientProtocol,
            cost: CostServiceProtocol,
            artifacts: ArtifactsServiceProtocol
    ):
        self.llm = llm
        self.cost = cost
        self.artifacts = artifacts

    async def ask(self, prompt: str, prompt_system: str) -> str:
        try:
            # Calculate token count for prompt and system prompt
            prompt_tokens = count_tokens(prompt)
            system_tokens = count_tokens(prompt_system)
            total_tokens = prompt_tokens + system_tokens

            logger.debug(f"LLM Prompt System: {prompt_system}")
            logger.debug(f"LLM Prompt: {prompt}")
            logger.debug(f"Prompt token count: prompt={prompt_tokens}, system={system_tokens}, total={total_tokens}")

            # Check if prompt exceeds maximum token limit
            max_prompt_tokens = settings.llm.meta.max_prompt_tokens
            if max_prompt_tokens is not None and total_tokens > max_prompt_tokens:
                logger.warning(
                    f"Prompt exceeds maximum token limit: {total_tokens} > {max_prompt_tokens}"
                )

            await hook.emit_chat_start(prompt, prompt_system)
            result = await self.llm.chat(prompt, prompt_system)
            logger.debug(f"LLM Response: {result.text}")
            if not result.text:
                logger.warning(
                    f"LLM returned an empty response (prompt length={len(prompt)} chars, tokens={total_tokens})"
                )

            report = self.cost.calculate(result)
            if report:
                logger.info(report.pretty())

            await hook.emit_chat_complete(result, report)
            await self.artifacts.save_llm(
                prompt=prompt,
                response=result.text,
                cost_report=report,
                prompt_system=prompt_system,
            )

            return result.text
        except Exception as error:
            logger.exception(f"LLM request failed: {error}")
            await hook.emit_chat_error(prompt, prompt_system)
