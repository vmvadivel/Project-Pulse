"""
Compression middleware for FastAPI RAG application.
Provides gzip compression for responses to reduce bandwidth and improve latency.
"""

import gzip
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.datastructures import Headers, MutableHeaders

logger = logging.getLogger(__name__)


class GZipMiddleware(BaseHTTPMiddleware):
    """
    Custom GZip compression middleware with configurable settings.
    
    Compresses response bodies when:
    1. Client accepts gzip encoding (Accept-Encoding: gzip)
    2. Response size exceeds minimum threshold
    3. Content-Type is compressible
    
    Features:
    - Configurable minimum size threshold
    - Configurable compression level
    - Automatic content-type detection
    - Proper header management
    - Metrics tracking
    """
    
    def __init__(
        self,
        app,
        minimum_size: int = 500,  # Only compress responses > 500 bytes
        compression_level: int = 6,  # 1 (fastest) to 9 (best compression)
        compressible_types: list = None
    ):
        """
        Initialize GZip middleware.
        
        Args:
            app: FastAPI application
            minimum_size: Minimum response size (bytes) to compress
            compression_level: Gzip compression level (1-9)
            compressible_types: List of content types to compress
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        
        # Default compressible content types
        if compressible_types is None:
            self.compressible_types = {
                'text/html',
                'text/plain',
                'text/css',
                'text/javascript',
                'application/javascript',
                'application/json',
                'application/xml',
                'text/xml',
                'application/x-javascript',
            }
        else:
            self.compressible_types = set(compressible_types)
        
        # Track compression stats
        self.total_requests = 0
        self.compressed_requests = 0
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0
        
        logger.info(
            f"GZip middleware initialized: "
            f"min_size={minimum_size}B, "
            f"level={compression_level}"
        )
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request and compress response if appropriate.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint handler
            
        Returns:
            Response (compressed or original)
        """
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        
        # Get response from endpoint
        response = await call_next(request)
        
        # Don't compress if client doesn't accept gzip
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Don't compress if already compressed
        if response.headers.get("content-encoding"):
            return response
        
        # Don't compress streaming responses
        if isinstance(response, StreamingResponse):
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        original_size = len(body)
        self.total_requests += 1
        self.total_bytes_original += original_size
        
        # Check if response is large enough to compress
        if original_size < self.minimum_size:
            # Return original response
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Check if content type is compressible
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in self.compressible_types:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress the response
        try:
            compressed_body = gzip.compress(
                body,
                compresslevel=self.compression_level
            )
            compressed_size = len(compressed_body)
            
            # Only use compression if it actually reduces size
            if compressed_size < original_size:
                self.compressed_requests += 1
                self.total_bytes_compressed += compressed_size
                
                # Calculate compression ratio
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                # Log compression stats (debug level)
                logger.debug(
                    f"Compressed {request.url.path}: "
                    f"{original_size}B → {compressed_size}B "
                    f"({compression_ratio:.1f}% reduction)"
                )
                
                # Create new headers
                headers = MutableHeaders(response.headers)
                headers["content-encoding"] = "gzip"
                headers["content-length"] = str(compressed_size)
                
                # Remove any existing content-length header
                if "vary" in headers:
                    if "accept-encoding" not in headers["vary"].lower():
                        headers["vary"] = f"{headers['vary']}, Accept-Encoding"
                else:
                    headers["vary"] = "Accept-Encoding"
                
                return Response(
                    content=compressed_body,
                    status_code=response.status_code,
                    headers=dict(headers),
                    media_type=response.media_type
                )
            else:
                # Compression made it bigger, return original
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
                
        except Exception as e:
            logger.error(f"Error compressing response: {e}")
            # Return original response on compression error
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
    
    def get_stats(self) -> dict:
        """
        Get compression statistics.
        
        Returns:
            Dict with compression metrics
        """
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "compressed_requests": 0,
                "compression_rate": 0.0,
                "bytes_saved": 0,
                "bandwidth_reduction": 0.0
            }
        
        bytes_saved = self.total_bytes_original - self.total_bytes_compressed
        bandwidth_reduction = (bytes_saved / self.total_bytes_original * 100) if self.total_bytes_original > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "compressed_requests": self.compressed_requests,
            "compression_rate": (self.compressed_requests / self.total_requests * 100),
            "original_bytes": self.total_bytes_original,
            "compressed_bytes": self.total_bytes_compressed,
            "bytes_saved": bytes_saved,
            "bandwidth_reduction": bandwidth_reduction
        }