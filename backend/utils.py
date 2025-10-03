"""
Utility functions for RAG System.
Includes file validation, formatting helpers, retry logic, and rate limiting (for future use).
"""

import os
import time
import math
from typing import Optional
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
from fastapi import UploadFile

from config import (
    MAX_FILE_SIZE, 
    SUPPORTED_EXTENSIONS, 
    FILE_TYPE_MAP
)
from exceptions import (
    UnsupportedFileType,
    FileTooLarge,
    FileAlreadyExists,
    LLMRateLimitExceeded
)


# ============================================================================
# File Validation
# ============================================================================

def get_file_type_from_extension(filename: str) -> str:
    """Get file type from file extension."""
    ext = os.path.splitext(filename)[1].lower()
    return FILE_TYPE_MAP.get(ext, 'Unknown')


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def validate_uploaded_file(
    file: UploadFile, 
    existing_files: set,
    max_size: int = MAX_FILE_SIZE
) -> None:
    """
    Validate uploaded file before processing.
    
    Args:
        file: The uploaded file
        existing_files: Set of filenames already in the system
        max_size: Maximum allowed file size in bytes
        
    Raises:
        FileTooLarge: If file exceeds max size
        UnsupportedFileType: If file type not supported
        FileAlreadyExists: If filename already exists
    """
    # Check file size
    if file.size and file.size > max_size:
        raise FileTooLarge(file.filename, file.size, max_size)
    
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        file_type = get_file_type_from_extension(file.filename)
        raise UnsupportedFileType(file.filename, file_type)
    
    # Check if file already exists
    if file.filename in existing_files:
        raise FileAlreadyExists(file.filename)


# ============================================================================
# Retry Logic with Exponential Backoff
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
        
    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def unreliable_api_call():
            # Code that might fail
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    import random
                    delay = delay * (0.5 + random.random())
                    
                    print(f"Retry {retries}/{max_retries} after {delay:.2f}s due to: {str(e)}")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator


# ============================================================================
# Simple In-Memory Rate Limiter (for demo/portfolio)
# ============================================================================

class SimpleRateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    Suitable for single-instance demo deployments.
    
    For production with multiple instances, use Redis-backed rate limiting.
    """
    
    def __init__(self):
        self._requests = defaultdict(list)
        self._lock = Lock()
    
    def is_allowed(
        self, 
        identifier: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP address, user ID, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        with self._lock:
            now = time.time()
            window_start = now - window_seconds
            
            # Get requests for this identifier
            requests = self._requests[identifier]
            
            # Remove requests outside the window
            requests[:] = [req_time for req_time in requests if req_time > window_start]
            
            # Check if under limit
            if len(requests) < max_requests:
                requests.append(now)
                return True, None
            else:
                # Calculate when the oldest request will expire
                oldest_request = requests[0]
                retry_after = int(oldest_request + window_seconds - now) + 1
                return False, retry_after
    
    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """
        Clean up old entries to prevent memory growth.
        Should be called periodically.
        
        Args:
            max_age_seconds: Remove entries older than this
        """
        with self._lock:
            now = time.time()
            cutoff = now - max_age_seconds
            
            # Remove identifiers with no recent requests
            identifiers_to_remove = []
            for identifier, requests in self._requests.items():
                if not requests or all(req_time < cutoff for req_time in requests):
                    identifiers_to_remove.append(identifier)
            
            for identifier in identifiers_to_remove:
                del self._requests[identifier]


# Global rate limiter instance
rate_limiter = SimpleRateLimiter()


def check_rate_limit(
    identifier: str,
    endpoint: str,
    max_requests: int = 100,
    window_seconds: int = 3600
) -> None:
    """
    Check rate limit for an identifier and endpoint.
    
    Args:
        identifier: Unique identifier (IP address, etc.)
        endpoint: Endpoint name (for different limits per endpoint)
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        
    Raises:
        LLMRateLimitExceeded: If rate limit is exceeded
    """
    key = f"{identifier}:{endpoint}"
    allowed, retry_after = rate_limiter.is_allowed(key, max_requests, window_seconds)
    
    if not allowed:
        raise LLMRateLimitExceeded(
            provider=f"Demo Rate Limit ({endpoint})",
            retry_after=retry_after or 60
        )


# ============================================================================
# Time and Date Formatting
# ============================================================================

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp in ISO format."""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.isoformat()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ============================================================================
# String Sanitization
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace problematic characters
    invalid_chars = '<>:"|?*\\'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return name + ext


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# Data Validation Helpers
# ============================================================================

def is_valid_query(query: str, min_length: int = 1, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate query string.
    
    Args:
        query: Query string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"
    
    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    return True, None


# ============================================================================
# Statistics Helpers
# ============================================================================

def calculate_percentile(values: list, percentile: float) -> float:
    """
    Calculate percentile of a list of values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        fraction = index - int(index)
        return lower + (upper - lower) * fraction


def calculate_average(values: list) -> float:
    """Calculate average of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)