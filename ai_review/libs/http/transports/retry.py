import asyncio
from http import HTTPStatus
from typing import TYPE_CHECKING

from httpx import Request, Response, AsyncBaseTransport, ReadTimeout, ConnectTimeout, NetworkError

if TYPE_CHECKING:
    from loguru import Logger


class RetryTransport(AsyncBaseTransport):
    def __init__(
            self,
            logger: "Logger",
            transport: AsyncBaseTransport,
            max_retries: int = 5,
            retry_delay: float = 0.5,
            retry_status_codes: tuple[HTTPStatus, ...] = (
                    HTTPStatus.BAD_GATEWAY,
                    HTTPStatus.GATEWAY_TIMEOUT,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            retry_exceptions: tuple[type[Exception], ...] = (
                    ReadTimeout,
                    ConnectTimeout,
                    NetworkError,
            )
    ):
        self.logger = logger
        self.transport = transport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_status_codes = retry_status_codes
        self.retry_exceptions = retry_exceptions

    async def handle_async_request(self, request: Request) -> Response:
        last_response: Response | None = None
        for attempt in range(self.max_retries):
            try:
                last_response = await self.transport.handle_async_request(request)
                if last_response.status_code not in self.retry_status_codes:
                    return last_response

                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed "
                    f"with status={last_response.status_code} for {request.method} {request.url}. "
                    f"Retrying in {self.retry_delay:.1f}s..."
                )
            except Exception as e:
                if isinstance(e, self.retry_exceptions):
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed "
                        f"with exception: {type(e).__name__} - {str(e)} for {request.method} {request.url}. "
                        f"Retrying in {self.retry_delay:.1f}s..."
                    )
                else:
                    raise e

            await asyncio.sleep(self.retry_delay)

        if last_response:
            self.logger.error(
                f"All {self.max_retries} attempts failed for "
                f"{request.method} {request.url} (last status={last_response.status_code})"
            )
            return last_response
        else:
            raise Exception(f"All {self.max_retries} attempts failed for {request.method} {request.url}")
