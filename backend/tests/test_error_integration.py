"""
Integration test script to verify error handling works correctly.
Run this with the server running: uvicorn main:app --reload

Usage: python test_error_integration.py
"""

import requests
import json
import io
from pathlib import Path

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'

test_count = 0
pass_count = 0


def run_test(description, test_func, expected_in_response):
    """Run a single test and check if expected string is in response."""
    global test_count, pass_count
    test_count += 1
    
    print(f"{Colors.YELLOW}Test {test_count}: {description}{Colors.NC}")
    
    try:
        response = test_func()
        response_text = json.dumps(response) if isinstance(response, dict) else str(response)
        
        if expected_in_response.lower() in response_text.lower():
            print(f"{Colors.GREEN}✓ PASS{Colors.NC}")
            pass_count += 1
            return True
        else:
            print(f"{Colors.RED}✗ FAIL{Colors.NC}")
            print(f"Expected to find: {expected_in_response}")
            print(f"Got: {response_text[:200]}")
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.NC}")
        return False
    finally:
        print()


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()


def test_unsupported_file():
    """Test uploading unsupported file type."""
    # Create a fake exe file
    fake_exe = io.BytesIO(b"fake executable content")
    files = {'file': ('malware.exe', fake_exe, 'application/octet-stream')}
    
    response = requests.post(f"{BASE_URL}/ingest", files=files)
    return response.json()


def test_query_empty_kb():
    """Test querying empty knowledge base."""
    # First clear all files
    requests.delete(f"{BASE_URL}/files")
    
    # Then try to query
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": "test query"}
    )
    return response.json()


def test_delete_nonexistent():
    """Test deleting non-existent file."""
    response = requests.delete(f"{BASE_URL}/files/nonexistent_file.pdf")
    return response.json()


def test_invalid_request():
    """Test sending invalid request format."""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"invalid_field": "data"}
    )
    return response.json()


def test_duplicate_file():
    """Test uploading the same file twice."""
    # Create a small test file
    test_content = b"This is a test document for duplicate testing."
    files = {'file': ('test_duplicate.txt', io.BytesIO(test_content), 'text/plain')}
    
    # Upload first time (should succeed)
    response1 = requests.post(f"{BASE_URL}/ingest", files=files)
    
    # Upload second time (should fail with FILE_005)
    files2 = {'file': ('test_duplicate.txt', io.BytesIO(test_content), 'text/plain')}
    response2 = requests.post(f"{BASE_URL}/ingest", files=files2)
    
    # Clean up
    requests.delete(f"{BASE_URL}/files/test_duplicate.txt")
    
    return response2.json()


def test_file_too_large():
    """Test uploading file that exceeds size limit (simulated)."""
    # Note: This is hard to test without actually creating a large file
    # In real scenario, you'd create a file > 100MB
    # For now, we'll skip this test
    return {"skipped": "Manual test required for file size limit"}


def main():
    print("=" * 60)
    print("Testing RAG System Error Handling")
    print("=" * 60)
    print()
    
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}ERROR: Server not running at {BASE_URL}{Colors.NC}")
        print("Please start the server: uvicorn main:app --reload")
        return
    
    print("1. Testing Health Endpoint")
    print("-" * 60)
    run_test(
        "Health check should return healthy status",
        test_health,
        "healthy"
    )
    
    print("2. Testing File Upload Errors")
    print("-" * 60)
    run_test(
        "Upload unsupported file type (.exe) should return FILE_001",
        test_unsupported_file,
        "FILE_001"
    )
    
    run_test(
        "Upload duplicate file should return FILE_005",
        test_duplicate_file,
        "FILE_005"
    )
    
    print("3. Testing Chat Errors")
    print("-" * 60)
    run_test(
        "Query empty knowledge base should return RETRIEVAL_001",
        test_query_empty_kb,
        "RETRIEVAL_001"
    )
    
    print("4. Testing File Management Errors")
    print("-" * 60)
    run_test(
        "Delete non-existent file should return 404 or error",
        test_delete_nonexistent,
        "not found"
    )
    
    print("5. Testing Validation Errors")
    print("-" * 60)
    run_test(
        "Invalid request format should return validation error",
        test_invalid_request,
        "validation"
    )
    
    # Summary
    print()
    print("=" * 60)
    print(f"Test Results: {pass_count}/{test_count} passed")
    print("=" * 60)
    
    if pass_count == test_count:
        print(f"{Colors.GREEN}All tests passed!{Colors.NC}")
        print("\n✓ Error handling is working correctly")
        print("✓ Ready to proceed with next steps")
    else:
        print(f"{Colors.RED}Some tests failed{Colors.NC}")
        print("\nPlease review the failures above and fix any issues")
        print("before proceeding to the next steps.")
    
    print()


if __name__ == "__main__":
    main()