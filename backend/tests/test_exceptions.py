"""
Quick test script to verify exceptions.py works correctly.
Run this BEFORE integrating into main.py to catch any issues.

Usage: python test_exceptions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import *
import json


def test_exception_creation():
    """Test that all exceptions can be created without errors"""
    print("Testing exception creation...")
    
    exceptions_to_test = [
        MissingAPIKey(),
        InvalidAPIKey(),
        InvalidConfiguration("test_config", "test reason"),
        UnsupportedFileType("test.exe", "executable"),
        FileTooLarge("large.pdf", 150_000_000, 100_000_000),
        FileCorrupted("bad.pdf", "Password protected"),
        FileProcessingError("test.pdf", "Parser error"),
        FileAlreadyExists("duplicate.pdf"),
        QdrantConnectionFailed("./qdrant_storage", "Timeout"),
        QdrantUpsertFailed("test_collection", 100, "Connection lost"),
        CollectionNotFound("missing_collection"),
        VectorStoreSyncError("upsert", "Network error"),
        LLMServiceUnavailable("Groq", "API down"),
        LLMRateLimitExceeded("Groq", 60),
        LLMInvalidResponse("Malformed JSON"),
        LLMTimeout(30),
        NoDocumentsIngested(),
        RetrievalFailed("test query", "No results"),
        InsufficientContext("test query"),
        InvalidQuery("Empty query"),
        InvalidFileMetadata("filename", "Invalid characters"),
        InvalidRequest("Missing parameters"),
    ]
    
    for exc in exceptions_to_test:
        try:
            # check that exception has required attributes
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'error_code')
            assert hasattr(exc, 'status_code')
            assert hasattr(exc, 'details')
            assert hasattr(exc, 'timestamp')
            
            # test that to_dict() works
            error_dict = exc.to_dict()
            assert 'error' in error_dict
            assert 'type' in error_dict['error']
            assert 'message' in error_dict['error']
            assert 'code' in error_dict['error']
            
            print(f"✓ {exc.__class__.__name__} - OK")
        except Exception as e:
            print(f"✗ {exc.__class__.__name__} - FAILED: {e}")
            return False
    
    print("\n✓ All exceptions created successfully!\n")
    return True


def test_exception_serialization():
    """Test that exceptions can be serialized to JSON"""
    print("Testing JSON serialization...")
    
    exc = UnsupportedFileType("malware.exe", "executable")
    error_dict = exc.to_dict("test-request-123")
    
    try:
        # try to serialize to JSON
        json_str = json.dumps(error_dict, indent=2)
        print("Sample error response:")
        print(json_str)
        print("\n✓ JSON serialization works!\n")
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


def test_status_codes():
    """Verify that status codes are correct"""
    print("Testing HTTP status codes...")
    
    status_code_tests = [
        (MissingAPIKey(), 500, "CONFIG errors should be 500"),
        (UnsupportedFileType("test.exe", "exe"), 400, "Unsupported file should be 400"),
        (FileTooLarge("big.pdf", 200_000_000, 100_000_000), 413, "Too large should be 413"),
        (FileCorrupted("bad.pdf"), 422, "Corrupted file should be 422"),
        (QdrantConnectionFailed("./storage"), 503, "Qdrant down should be 503"),
        (LLMRateLimitExceeded(), 429, "Rate limit should be 429"),
        (LLMTimeout(), 504, "Timeout should be 504"),
        (NoDocumentsIngested(), 400, "No docs should be 400"),
    ]
    
    all_passed = True
    for exc, expected_code, description in status_code_tests:
        if exc.status_code == expected_code:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - Expected {expected_code}, got {exc.status_code}")
            all_passed = False
    
    print()
    return all_passed


def test_error_details():
    """Verify that error details contain helpful information"""
    print("Testing error details...")
    
    exc = UnsupportedFileType("document.exe", "executable")
    error_dict = exc.to_dict()
    
    details = error_dict['error']['details']
    
    checks = [
        ('filename' in details, "Should include filename"),
        ('supported_types' in details, "Should list supported types"),
        ('suggestion' in details, "Should provide helpful suggestion"),
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✓ {description}")
        else:
            print(f"✗ {description}")
            all_passed = False
    
    print()
    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Testing exceptions.py")
    print("=" * 60)
    print()
    
    tests = [
        test_exception_creation,
        test_exception_serialization,
        test_status_codes,
        test_error_details,
    ]
    
    results = [test() for test in tests]
    
    print("=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("exceptions.py is ready to be integrated into main.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please fix the issues before integrating")
    print("=" * 60)