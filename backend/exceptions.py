from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uuid

import logging
logger = logging.getLogger(__name__)


class BaseRAGException(Exception):
    """Base exception for all RAG system errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self, request_id: str = None) -> Dict[str, Any]:
        """Convert exception to dict for JSON response"""
        return {
            "error": {
                "type": self.__class__.__name__,
                "message": self.message,
                "code": self.error_code,
                "details": self.details,
                "timestamp": self.timestamp,
                "request_id": request_id or str(uuid.uuid4())
            }
        }


# Configuration Errors (500)
class MissingAPIKey(BaseRAGException):
    """Raised when required API key is not found in environment"""
    
    def __init__(self, service: str = "Groq LLM"):
        super().__init__(
            message="System configuration error",
            error_code="CONFIG_001",
            details={
                "service": service,
                "suggestion": "This is a configuration issue. Please contact the administrator."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class InvalidAPIKey(BaseRAGException):
    """API key is invalid or rejected by service"""
    
    def __init__(self, service: str = "Groq LLM"):
        super().__init__(
            message="Authentication failed with AI service",
            error_code="CONFIG_002",
            details={
                "provider": service,
                "suggestion": "This is a configuration issue. The administrator has been notified."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class InvalidConfiguration(BaseRAGException):
    """System configuration is invalid"""
    
    def __init__(self, config_item: str, reason: str):
        super().__init__(
            message=f"Invalid configuration: {config_item}",
            error_code="CONFIG_003",
            details={
                "config_item": config_item,
                "reason": reason,
                "suggestion": "Please check system configuration."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# File Errors (400, 413, 422)

class UnsupportedFileType(BaseRAGException):
    """Raised when uploaded file type is not supported"""
    
    def __init__(self, filename: str, detected_type: str = "unknown"):
        supported_types = ["pdf", "txt", "doc", "docx", "csv", "xlsx", "pptx", "html", "md", "json", "xml"]
        super().__init__(
            message="File type not supported for processing",
            error_code="FILE_001",
            details={
                "filename": filename,
                "detected_type": detected_type,
                "supported_types": supported_types,
                "suggestion": "Please upload a document in a supported format"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )


class FileTooLarge(BaseRAGException):
    """File exceeds size limit"""
    
    def __init__(self, filename: str, size_bytes: int, max_size_bytes: int):
        super().__init__(
            message="File size exceeds maximum allowed",
            error_code="FILE_002",
            details={
                "filename": filename,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "max_size_mb": round(max_size_bytes / (1024 * 1024), 2),
                "suggestion": f"Please upload a file smaller than {round(max_size_bytes / (1024 * 1024))}MB"
            },
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        )


class FileCorrupted(BaseRAGException):
    """File cannot be parsed or is corrupted"""
    
    def __init__(self, filename: str, reason: str = "Unknown"):
        super().__init__(
            message="Unable to read file contents",
            error_code="FILE_003",
            details={
                "filename": filename,
                "reason": reason,
                "suggestion": "Please ensure the file is valid and not password-protected"
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class FileProcessingError(BaseRAGException):
    """Raised when file processing fails"""
    
    def __init__(self, filename: str, error: str):
        super().__init__(
            message="Error processing file",
            error_code="FILE_004",
            details={
                "filename": filename,
                "error": error,
                "suggestion": "Please try uploading again. If the problem persists, try a different file format."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class FileAlreadyExists(BaseRAGException):
    """File with this name already exists"""
    
    def __init__(self, filename: str):
        super().__init__(
            message="File already exists in knowledge base",
            error_code="FILE_005",
            details={
                "filename": filename,
                "suggestion": "Please use a different filename or delete the existing file first"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )



# Vector Store Errors (500, 503)

class QdrantConnectionFailed(BaseRAGException):
    """Connection to Qdrant failed"""
    
    def __init__(self, storage_path: str, reason: str = "Connection timeout"):
        super().__init__(
            message="Vector database temporarily unavailable",
            error_code="VSTORE_001",
            details={
                "storage_path": storage_path,
                "reason": reason,
                "retry_after": 30,
                "suggestion": "Please try again in a few moments"
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class QdrantUpsertFailed(BaseRAGException):
    """Failed to upsert documents to Qdrant"""
    
    def __init__(self, collection_name: str, num_points: int, error: str):
        super().__init__(
            message="Failed to store documents in vector database",
            error_code="VSTORE_002",
            details={
                "collection": collection_name,
                "num_points": num_points,
                "error": error,
                "suggestion": "Please try uploading again"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class CollectionNotFound(BaseRAGException):
    """Qdrant collection doesn't exist"""
    
    def __init__(self, collection_name: str):
        super().__init__(
            message="Vector database collection not found",
            error_code="VSTORE_003",
            details={
                "collection": collection_name,
                "suggestion": "This is a system error. Please contact the administrator."
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class VectorStoreSyncError(BaseRAGException):
    """Vector store synchronization failed"""
    
    def __init__(self, operation: str, error: str):
        super().__init__(
            message="Vector database synchronization failed",
            error_code="VSTORE_004",
            details={
                "operation": operation,
                "error": error,
                "suggestion": "Please try the operation again"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# LLM Errors

class LLMServiceUnavailable(BaseRAGException):
    """LLM service is unavailable"""
    
    def __init__(self, provider: str = "Groq", reason: str = "Service unavailable"):
        super().__init__(
            message="AI service temporarily unavailable",
            error_code="LLM_001",
            details={
                "provider": provider,
                "reason": reason,
                "retry_after": 60,
                "suggestion": "Please try again in a few moments"
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class LLMRateLimitExceeded(BaseRAGException):
    """LLM API rate limit exceeded"""
    
    def __init__(self, provider: str = "Groq", retry_after: int = 60):
        super().__init__(
            message="AI service rate limit exceeded",
            error_code="LLM_002",
            details={
                "provider": provider,
                "retry_after": retry_after,
                "suggestion": f"The demo has reached its rate limit. Please try again in {retry_after} seconds"
            },
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )


class LLMInvalidResponse(BaseRAGException):
    """LLM returned invalid or unexpected response"""
    
    def __init__(self, error: str):
        super().__init__(
            message="Received invalid response from AI service",
            error_code="LLM_003",
            details={
                "error": error,
                "suggestion": "Please try your query again"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class LLMTimeout(BaseRAGException):
    """LLM request timed out"""
    
    def __init__(self, timeout_seconds: int = 30):
        super().__init__(
            message="AI service request timed out",
            error_code="LLM_004",
            details={
                "timeout": timeout_seconds,
                "suggestion": "Please try with a shorter or simpler query"
            },
            status_code=status.HTTP_504_GATEWAY_TIMEOUT
        )


class NoDocumentsIngested(BaseRAGException):
    """User tried to query empty knowledge base"""
    
    def __init__(self):
        super().__init__(
            message="No documents available to search",
            error_code="RETRIEVAL_001",
            details={
                "suggestion": "Please upload at least one document using the /ingest endpoint before querying"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )


class RetrievalFailed(BaseRAGException):
    """Document retrieval failed"""
    
    def __init__(self, query: str, error: str):
        super().__init__(
            message="Failed to retrieve relevant documents",
            error_code="RETRIEVAL_002",
            details={
                "query": query[:100],  
                "error": error,
                "suggestion": "Please try rephrasing your query"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class InsufficientContext(BaseRAGException):
    """Retrieved context is insufficient to answer query"""
    
    def __init__(self, query: str):
        super().__init__(
            message="Insufficient information found to answer query",
            error_code="RETRIEVAL_003",
            details={
                "query": query[:100],
                "suggestion": "Try rephrasing your question or upload more relevant documents"
            },
            status_code=status.HTTP_200_OK 
        )


# Validation Errors (400)

class InvalidQuery(BaseRAGException):
    """Query is invalid"""
    
    def __init__(self, reason: str):
        super().__init__(
            message="Invalid query",
            error_code="VALIDATION_001",
            details={
                "reason": reason,
                "suggestion": "Please check your query and try again"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )


class InvalidFileMetadata(BaseRAGException):
    """File metadata is invalid"""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            message="Invalid file metadata",
            error_code="VALIDATION_002",
            details={
                "field": field,
                "reason": reason,
                "suggestion": "Please check the file metadata"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )


class InvalidRequest(BaseRAGException):
    """Request is invalid"""
    
    def __init__(self, reason: str):
        super().__init__(
            message="Invalid request",
            error_code="VALIDATION_003",
            details={
                "reason": reason,
                "suggestion": "Please check your request parameters"
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )


# Error handler registration
def register_error_handlers(app):
    """Register all error handlers with FastAPI app"""
    
    @app.exception_handler(BaseRAGException)
    async def rag_exception_handler(request: Request, exc: BaseRAGException):
        """Handle custom RAG exceptions"""
        request_id = str(uuid.uuid4())
        
        logger.error(
            f"{exc.error_code} - {exc.message} | Request ID: {request_id}",
            extra={'error_code': exc.error_code, 'request_id': request_id}
        )
        if exc.details:
            logger.error(f"Error details: {exc.details}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(request_id)
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI validation errors"""
        request_id = str(uuid.uuid4())
        
        # extract field-specific errors
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        logger.warning(
            f"Request validation failed | Request ID: {request_id}",
            extra={'request_id': request_id, 'validation_errors': errors}
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "code": "VALIDATION_000",
                    "details": {
                        "errors": errors,
                        "suggestion": "Please check the request format and required fields"
                    },
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all for unexpected exceptions"""
        request_id = str(uuid.uuid4())
        
        logger.critical(
            f"UNHANDLED EXCEPTION | Request ID: {request_id} | Type: {type(exc).__name__}",
            exc_info=True,
            extra={'request_id': request_id}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "details": {
                        "suggestion": "Please try again. If the problem persists, contact support.",
                        "error_type": type(exc).__name__  
                    },
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
            }
        )
    
    logger.info("Error handlers registered successfully")