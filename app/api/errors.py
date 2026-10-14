from typing import Any, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.errors")


class AppException(Exception):
    """Base application exception with HTTP status code and error details."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class DocumentEmptyError(AppException):
    def __init__(self, message: str = "Uploaded file is empty or contains no extractable text", details: Optional[Any] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="DOCUMENT_EMPTY",
            message=message,
            details=details,
        )


class FileSizeLimitError(AppException):
    def __init__(self, max_size_mb: int, details: Optional[Any] = None):
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            code="FILE_SIZE_LIMIT_EXCEEDED",
            message=f"File exceeds maximum allowed size of {max_size_mb} MB",
            details=details,
        )


class UnsupportedMediaTypeError(AppException):
    def __init__(self, extension: str, allowed: list[str]):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Unsupported file extension '{extension}'. Allowed extensions: {', '.join(allowed)}",
        )


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="DOCUMENT_NOT_FOUND",
            message=f"Document '{document_id}' was not found",
        )


class InvalidSessionIdError(AppException):
    def __init__(self, message: str = "Header 'X-Session-ID' must be a valid UUID", details: Optional[Any] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_SESSION_ID",
            message=message,
            details=details,
        )


class SessionLimitExceededError(AppException):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            code="SESSION_LIMIT_EXCEEDED",
            message=message,
            details=details,
        )


class VectorStoreNotReadyError(AppException):
    def __init__(self, message: str = "Vector index is currently unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="VECTOR_STORE_UNAVAILABLE",
            message=message,
        )


class LLMProviderError(AppException):
    def __init__(self, message: str = "LLM provider failed or timed out", details: Optional[Any] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="LLM_PROVIDER_ERROR",
            message=message,
            details=details,
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.warning(f"Handled application exception: [{exc.code}] {exc.message} on {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters or payload",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled server error on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please consult server logs.",
                    "details": None,
                }
            },
        )
