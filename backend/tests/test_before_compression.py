"""
Test script to measure API performance WITHOUT gzip compression.
Run this first: uvicorn main:app --reload

Usage: python tests/test_before_compression.py
"""

import requests
import time
import json
import io
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from statistics import mean, median

BASE_URL = "http://localhost:8000"

class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'


class PerformanceMetrics:
    """Track performance metrics for each endpoint"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.response_sizes = []
        self.latencies = []
        self.transfer_times = []
        self.requests_made = 0
    
    def record(self, size_bytes: int, latency_ms: float, transfer_time_ms: float):
        """Record a single request metric"""
        self.response_sizes.append(size_bytes)
        self.latencies.append(latency_ms)
        self.transfer_times.append(transfer_time_ms)
        self.requests_made += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.response_sizes:
            return None
        
        return {
            'endpoint': self.endpoint,
            'requests': self.requests_made,
            'avg_size_bytes': int(mean(self.response_sizes)),
            'avg_size_kb': round(mean(self.response_sizes) / 1024, 2),
            'total_size_kb': round(sum(self.response_sizes) / 1024, 2),
            'avg_latency_ms': round(mean(self.latencies), 2),
            'median_latency_ms': round(median(self.latencies), 2),
            'avg_transfer_time_ms': round(mean(self.transfer_times), 2),
            'median_transfer_time_ms': round(median(self.transfer_times), 2),
        }


def format_size(bytes_val: int) -> str:
    """Format bytes to human readable"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.2f} MB"


def make_request(url: str, method: str = 'GET', **kwargs) -> Tuple[requests.Response, float, float, int]:
    """
    Make HTTP request and measure metrics
    
    Returns:
        (response, latency_ms, transfer_time_ms, size_bytes)
    """
    # Ensure no compression in request headers
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    
    # Explicitly disable compression by not sending Accept-Encoding
    # or sending identity encoding
    kwargs['headers']['Accept-Encoding'] = 'identity'
    
    start_time = time.time()
    
    if method == 'GET':
        response = requests.get(url, **kwargs)
    elif method == 'POST':
        response = requests.post(url, **kwargs)
    elif method == 'DELETE':
        response = requests.delete(url, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    end_time = time.time()
    
    # Calculate metrics
    total_time_ms = (end_time - start_time) * 1000
    size_bytes = len(response.content)
    
    # Rough estimate: transfer time = total - (some processing overhead)
    # For simplicity, we'll use total time as transfer time
    transfer_time_ms = total_time_ms
    latency_ms = total_time_ms
    
    return response, latency_ms, transfer_time_ms, size_bytes


def test_endpoint(
    name: str,
    url: str,
    method: str = 'GET',
    iterations: int = 5,
    **kwargs
) -> PerformanceMetrics:
    """Test an endpoint multiple times and collect metrics"""
    
    print(f"\n{Colors.BLUE}Testing: {name}{Colors.NC}")
    print(f"Endpoint: {method} {url}")
    print(f"Iterations: {iterations}")
    print("-" * 60)
    
    metrics = PerformanceMetrics(name)
    
    for i in range(iterations):
        try:
            response, latency, transfer_time, size = make_request(url, method, **kwargs)
            
            if response.status_code == 200:
                metrics.record(size, latency, transfer_time)
                print(f"  Request {i+1}: {format_size(size)} in {latency:.0f}ms")
            else:
                print(f"  {Colors.RED}Request {i+1} failed: {response.status_code}{Colors.NC}")
        except Exception as e:
            print(f"  {Colors.RED}Request {i+1} error: {e}{Colors.NC}")
    
    # Show summary for this endpoint
    summary = metrics.get_summary()
    if summary:
        print(f"\n{Colors.GREEN}Summary:{Colors.NC}")
        print(f"  Avg Size: {format_size(summary['avg_size_bytes'])}")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"  Total Data: {summary['total_size_kb']:.2f} KB")
    
    return metrics


def create_sample_file(filename: str, size_kb: int = 10) -> io.BytesIO:
    """Create a sample text file for testing"""
    content = f"Sample document for testing - {filename}\n"
    content += "This is test content. " * (size_kb * 50)  # Rough size
    return io.BytesIO(content.encode('utf-8'))


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(BASE_URL, timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    print("="*60)
    print(f"{Colors.BOLD}RAG API Performance Test - WITHOUT Compression{Colors.NC}")
    print("="*60)
    print()
    
    # Check server
    if not check_server():
        print(f"{Colors.RED}ERROR: Server not running at {BASE_URL}{Colors.NC}")
        print("Please start the server: uvicorn main:app --reload")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Server is running{Colors.NC}\n")
    
    # Test configuration
    ITERATIONS = 5
    all_metrics: List[PerformanceMetrics] = []
    
    # 1. Test root endpoint
    metrics = test_endpoint(
        "Root Info",
        f"{BASE_URL}/",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 2. Test health endpoint
    metrics = test_endpoint(
        "Health Check",
        f"{BASE_URL}/health",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 3. Test stats endpoint
    metrics = test_endpoint(
        "System Stats",
        f"{BASE_URL}/stats",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 4. Upload a test file
    # print(f"\n{Colors.BLUE}Uploading test file...{Colors.NC}")
    # test_file = create_sample_file("test_doc.txt", size_kb=5)
    # files = {'file': ('test_doc.txt', test_file, 'text/plain')}
    
    # try:
    #    start = time.time()
    #    response, _, _, size = make_request(
    #        f"{BASE_URL}/ingest",
    #        method='POST',
    #        files=files
    #    )
    #    elapsed = (time.time() - start) * 1000
        
    #    if response.status_code == 200:
    #        print(f"{Colors.GREEN}✓ File uploaded: {format_size(size)} in {elapsed:.0f}ms{Colors.NC}")
    #    else:
    #        print(f"{Colors.RED}✗ Upload failed: {response.status_code}{Colors.NC}")
    #except Exception as e:
    #    print(f"{Colors.RED}✗ Upload error: {e}{Colors.NC}")
    
    # 5. Test files list endpoint (should have data now)
    metrics = test_endpoint(
        "File List",
        f"{BASE_URL}/files",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 6. Test chat endpoint
    test_queries = [
        #"What is this document about?",
        #"Summarize the content",
        #"Tell me more details"
        "Provide a comprehensive summary with all key details, examples, and quotes from the document",
        "List all major topics discussed in the document with explanations",
        "Give me a detailed analysis of the main arguments with supporting evidence"
    ]
    
    chat_metrics = PerformanceMetrics("Chat Queries")
    
    print(f"\n{Colors.BLUE}Testing: Chat Queries{Colors.NC}")
    print(f"Endpoint: POST {BASE_URL}/chat")
    print(f"Queries: {len(test_queries)}")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        try:
            response, latency, transfer_time, size = make_request(
                f"{BASE_URL}/chat",
                method='POST',
                json={"query": query}
            )
            
            if response.status_code == 200:
                chat_metrics.record(size, latency, transfer_time)
                print(f"  Query {i}: {format_size(size)} in {latency:.0f}ms")
            else:
                print(f"  {Colors.RED}Query {i} failed: {response.status_code}{Colors.NC}")
        except Exception as e:
            print(f"  {Colors.RED}Query {i} error: {e}{Colors.NC}")
    
    all_metrics.append(chat_metrics)
    
    # Show chat summary
    summary = chat_metrics.get_summary()
    if summary:
        print(f"\n{Colors.GREEN}Summary:{Colors.NC}")
        print(f"  Avg Size: {format_size(summary['avg_size_bytes'])}")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"  Total Data: {summary['total_size_kb']:.2f} KB")
    
    # Cleanup - delete test file
    print(f"\n{Colors.BLUE}Cleaning up...{Colors.NC}")
    try:
        response = requests.delete(f"{BASE_URL}/files/test_doc.txt")
        if response.status_code == 200:
            print(f"{Colors.GREEN}✓ Test file deleted{Colors.NC}")
    except:
        pass
    
    # Generate overall report
    print("\n" + "="*60)
    print(f"{Colors.BOLD}OVERALL RESULTS - WITHOUT COMPRESSION{Colors.NC}")
    print("="*60)
    
    total_requests = 0
    total_data_kb = 0
    avg_latencies = []
    
    print(f"\n{'Endpoint':<20} {'Requests':<10} {'Avg Size':<15} {'Avg Latency':<15}")
    print("-" * 60)
    
    for metric in all_metrics:
        summary = metric.get_summary()
        if summary:
            total_requests += summary['requests']
            total_data_kb += summary['total_size_kb']
            avg_latencies.append(summary['avg_latency_ms'])
            
            print(
                f"{summary['endpoint']:<20} "
                f"{summary['requests']:<10} "
                f"{summary['avg_size_kb']:>7.2f} KB    "
                f"{summary['avg_latency_ms']:>7.0f} ms"
            )
    
    print("-" * 60)
    print(f"\n{Colors.BOLD}Summary Statistics:{Colors.NC}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Total Data Transferred: {total_data_kb:.2f} KB")
    print(f"  Average Response Size: {total_data_kb/total_requests:.2f} KB")
    print(f"  Average Latency: {mean(avg_latencies):.0f} ms")
    print(f"  Median Latency: {median(avg_latencies):.0f} ms")
    
    # Save results to file
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'compression': False,
        'total_requests': total_requests,
        'total_data_kb': round(total_data_kb, 2),
        'avg_response_size_kb': round(total_data_kb/total_requests, 2),
        'avg_latency_ms': round(mean(avg_latencies), 2),
        'median_latency_ms': round(median(avg_latencies), 2),
        'endpoints': [m.get_summary() for m in all_metrics if m.get_summary()]
    }
    
    with open('results_before_compression.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Colors.GREEN}✓ Results saved to: results_before_compression.json{Colors.NC}")
    print("\n" + "="*60)
    print(f"{Colors.YELLOW}Next Steps:{Colors.NC}")
    print("1. Stop the current server")
    print("2. Copy middleware.py and main_with_compression.py")
    print("3. Start server: uvicorn main_with_compression:app --reload")
    print("4. Run: python test_after_compression.py")
    print("5. Run: python compare_compression_results.py")
    print("="*60)


if __name__ == "__main__":
    main()