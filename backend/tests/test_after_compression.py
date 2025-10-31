"""
Test script to measure API performance WITH gzip compression.
Need to have one old main.py file without gzip and the another one with gzip enabled
Run this second: uvicorn main_with_compression:app --reload

Usage: python tests/test_after_compression.py
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
        self.response_sizes_original = []  # Decompressed size
        self.response_sizes_compressed = []  # Compressed size (if applicable)
        self.latencies = []
        self.transfer_times = []
        self.requests_made = 0
        self.compressed_count = 0
    
    def record(self, original_size: int, compressed_size: int, latency_ms: float, 
               transfer_time_ms: float, was_compressed: bool):
        """Record a single request metric"""
        self.response_sizes_original.append(original_size)
        self.response_sizes_compressed.append(compressed_size)
        self.latencies.append(latency_ms)
        self.transfer_times.append(transfer_time_ms)
        self.requests_made += 1
        if was_compressed:
            self.compressed_count += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.response_sizes_original:
            return None
        
        return {
            'endpoint': self.endpoint,
            'requests': self.requests_made,
            'compressed_requests': self.compressed_count,
            'compression_rate': round((self.compressed_count / self.requests_made) * 100, 1),
            'avg_size_original_bytes': int(mean(self.response_sizes_original)),
            'avg_size_compressed_bytes': int(mean(self.response_sizes_compressed)),
            'avg_size_original_kb': round(mean(self.response_sizes_original) / 1024, 2),
            'avg_size_compressed_kb': round(mean(self.response_sizes_compressed) / 1024, 2),
            'total_size_original_kb': round(sum(self.response_sizes_original) / 1024, 2),
            'total_size_compressed_kb': round(sum(self.response_sizes_compressed) / 1024, 2),
            'bandwidth_saved_kb': round((sum(self.response_sizes_original) - sum(self.response_sizes_compressed)) / 1024, 2),
            'bandwidth_saved_percent': round((1 - sum(self.response_sizes_compressed) / sum(self.response_sizes_original)) * 100, 1),
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


def make_request(url: str, method: str = 'GET', **kwargs) -> Tuple[requests.Response, float, float, int, int, bool]:
    """
    Make HTTP request and measure metrics
    
    Returns:
        (response, latency_ms, transfer_time_ms, original_size_bytes, compressed_size_bytes, was_compressed)
    """
    # Enable compression by sending Accept-Encoding header
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    kwargs['headers']['Accept-Encoding'] = 'gzip, deflate'
    
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
    
    # Check if response was compressed
    was_compressed = response.headers.get('Content-Encoding') == 'gzip'
    
    # Get sizes
    original_size = len(response.content)  # Decompressed size
    
    # For compressed responses, get actual transfer size
    if was_compressed:
        # The raw.read() gives us the compressed size
        compressed_size = int(response.headers.get('Content-Length', len(response.content)))
    else:
        compressed_size = original_size
    
    transfer_time_ms = total_time_ms
    latency_ms = total_time_ms
    
    return response, latency_ms, transfer_time_ms, original_size, compressed_size, was_compressed


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
            response, latency, transfer_time, orig_size, comp_size, compressed = make_request(
                url, method, **kwargs
            )
            
            if response.status_code == 200:
                metrics.record(orig_size, comp_size, latency, transfer_time, compressed)
                
                comp_indicator = "🗜️ " if compressed else ""
                compression_ratio = (1 - comp_size/orig_size)*100 if compressed else 0
                
                print(f"  Request {i+1}: {comp_indicator}{format_size(comp_size)} ", end="")
                if compressed:
                    print(f"(saved {compression_ratio:.0f}%) ", end="")
                print(f"in {latency:.0f}ms")
            else:
                print(f"  {Colors.RED}Request {i+1} failed: {response.status_code}{Colors.NC}")
        except Exception as e:
            print(f"  {Colors.RED}Request {i+1} error: {e}{Colors.NC}")
    
    # Show summary for this endpoint
    summary = metrics.get_summary()
    if summary:
        print(f"\n{Colors.GREEN}Summary:{Colors.NC}")
        print(f"  Compressed: {summary['compressed_requests']}/{summary['requests']} ({summary['compression_rate']}%)")
        print(f"  Avg Original: {format_size(summary['avg_size_original_bytes'])}")
        print(f"  Avg Compressed: {format_size(summary['avg_size_compressed_bytes'])}")
        print(f"  Bandwidth Saved: {summary['bandwidth_saved_kb']:.2f} KB ({summary['bandwidth_saved_percent']:.1f}%)")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
    
    return metrics


def create_sample_file(filename: str, size_kb: int = 10) -> io.BytesIO:
    """Create a sample text file for testing"""
    content = f"Sample document for testing - {filename}\n"
    content += "This is test content. " * (size_kb * 50)  # Rough size
    return io.BytesIO(content.encode('utf-8'))


def check_server():
    """Check if server is running and compression is enabled"""
    try:
        response = requests.get(BASE_URL, timeout=2)
        if response.status_code != 200:
            return False, "Server returned non-200 status"
        
        # Check if compression feature is present
        data = response.json()
        features = data.get('features', [])
        if 'GZip Compression' not in features:
            return False, "GZip Compression not enabled (wrong server version?)"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    print("="*60)
    print(f"{Colors.BOLD}RAG API Performance Test - WITH Compression{Colors.NC}")
    print("="*60)
    print()
    
    # Check server
    server_ok, msg = check_server()
    if not server_ok:
        print(f"{Colors.RED}ERROR: {msg}{Colors.NC}")
        print("\nMake sure you're running the correct server:")
        print("  uvicorn main_with_compression:app --reload")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Server is running with compression enabled{Colors.NC}\n")
    
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
    
    # 4. Test compression stats endpoint (new)
    metrics = test_endpoint(
        "Compression Stats",
        f"{BASE_URL}/compression-stats",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 5. Upload a test file
    #print(f"\n{Colors.BLUE}Uploading test file...{Colors.NC}")
    #test_file = create_sample_file("test_doc.txt", size_kb=5)
    #files = {'file': ('test_doc.txt', test_file, 'text/plain')}
    
    #try:
    #    start = time.time()
    #    response, _, _, orig_size, comp_size, compressed = make_request(
    #        f"{BASE_URL}/ingest",
    #        method='POST',
    #        files=files
    #    )
    #    elapsed = (time.time() - start) * 1000
        
    #    if response.status_code == 200:
    #        comp_str = f" (compressed from {format_size(orig_size)})" if compressed else ""
    #        print(f"{Colors.GREEN}✓ File uploaded: {format_size(comp_size)}{comp_str} in {elapsed:.0f}ms{Colors.NC}")
    #    else:
    #        print(f"{Colors.RED}✗ Upload failed: {response.status_code}{Colors.NC}")
    #except Exception as e:
    #    print(f"{Colors.RED}✗ Upload error: {e}{Colors.NC}")
    
    # 6. Test files list endpoint (should have data now)
    metrics = test_endpoint(
        "File List",
        f"{BASE_URL}/files",
        iterations=ITERATIONS
    )
    all_metrics.append(metrics)
    
    # 7. Test chat endpoint
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
            response, latency, transfer_time, orig_size, comp_size, compressed = make_request(
                f"{BASE_URL}/chat",
                method='POST',
                json={"query": query}
            )
            
            if response.status_code == 200:
                chat_metrics.record(orig_size, comp_size, latency, transfer_time, compressed)
                
                comp_indicator = "🗜️ " if compressed else ""
                compression_ratio = (1 - comp_size/orig_size)*100 if compressed else 0
                
                print(f"  Query {i}: {comp_indicator}{format_size(comp_size)} ", end="")
                if compressed:
                    print(f"(saved {compression_ratio:.0f}%) ", end="")
                print(f"in {latency:.0f}ms")
            else:
                print(f"  {Colors.RED}Query {i} failed: {response.status_code}{Colors.NC}")
        except Exception as e:
            print(f"  {Colors.RED}Query {i} error: {e}{Colors.NC}")
    
    all_metrics.append(chat_metrics)
    
    # Show chat summary
    summary = chat_metrics.get_summary()
    if summary:
        print(f"\n{Colors.GREEN}Summary:{Colors.NC}")
        print(f"  Compressed: {summary['compressed_requests']}/{summary['requests']} ({summary['compression_rate']}%)")
        print(f"  Avg Original: {format_size(summary['avg_size_original_bytes'])}")
        print(f"  Avg Compressed: {format_size(summary['avg_size_compressed_bytes'])}")
        print(f"  Bandwidth Saved: {summary['bandwidth_saved_kb']:.2f} KB ({summary['bandwidth_saved_percent']:.1f}%)")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
    
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
    print(f"{Colors.BOLD}OVERALL RESULTS - WITH COMPRESSION{Colors.NC}")
    print("="*60)
    
    total_requests = 0
    total_compressed = 0
    total_data_original_kb = 0
    total_data_compressed_kb = 0
    avg_latencies = []
    
    print(f"\n{'Endpoint':<20} {'Compressed':<12} {'Orig Size':<12} {'Comp Size':<12} {'Saved %':<10} {'Latency':<10}")
    print("-" * 80)
    
    for metric in all_metrics:
        summary = metric.get_summary()
        if summary:
            total_requests += summary['requests']
            total_compressed += summary['compressed_requests']
            total_data_original_kb += summary['total_size_original_kb']
            total_data_compressed_kb += summary['total_size_compressed_kb']
            avg_latencies.append(summary['avg_latency_ms'])
            
            print(
                f"{summary['endpoint']:<20} "
                f"{summary['compressed_requests']}/{summary['requests']:<9} "
                f"{summary['avg_size_original_kb']:>7.2f} KB   "
                f"{summary['avg_size_compressed_kb']:>7.2f} KB   "
                f"{summary['bandwidth_saved_percent']:>5.1f}%     "
                f"{summary['avg_latency_ms']:>6.0f} ms"
            )
    
    print("-" * 80)
    
    bandwidth_saved_kb = total_data_original_kb - total_data_compressed_kb
    bandwidth_saved_pct = (bandwidth_saved_kb / total_data_original_kb * 100) if total_data_original_kb > 0 else 0
    
    print(f"\n{Colors.BOLD}Summary Statistics:{Colors.NC}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Compressed Requests: {total_compressed} ({total_compressed/total_requests*100:.1f}%)")
    print(f"  Original Data Size: {total_data_original_kb:.2f} KB")
    print(f"  Compressed Data Size: {total_data_compressed_kb:.2f} KB")
    print(f"  {Colors.GREEN}Bandwidth Saved: {bandwidth_saved_kb:.2f} KB ({bandwidth_saved_pct:.1f}%){Colors.NC}")
    print(f"  Average Latency: {mean(avg_latencies):.0f} ms")
    print(f"  Median Latency: {median(avg_latencies):.0f} ms")
    
    # Save results to file
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'compression': True,
        'total_requests': total_requests,
        'compressed_requests': total_compressed,
        'compression_rate': round((total_compressed/total_requests)*100, 1),
        'total_data_original_kb': round(total_data_original_kb, 2),
        'total_data_compressed_kb': round(total_data_compressed_kb, 2),
        'bandwidth_saved_kb': round(bandwidth_saved_kb, 2),
        'bandwidth_saved_percent': round(bandwidth_saved_pct, 1),
        'avg_response_size_original_kb': round(total_data_original_kb/total_requests, 2),
        'avg_response_size_compressed_kb': round(total_data_compressed_kb/total_requests, 2),
        'avg_latency_ms': round(mean(avg_latencies), 2),
        'median_latency_ms': round(median(avg_latencies), 2),
        'endpoints': [m.get_summary() for m in all_metrics if m.get_summary()]
    }
    
    with open('results_after_compression.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Colors.GREEN}✓ Results saved to: results_after_compression.json{Colors.NC}")
    print("\n" + "="*60)
    print(f"{Colors.YELLOW}Next Step:{Colors.NC}")
    print("Run: python compare_compression_results.py")
    print("="*60)


if __name__ == "__main__":
    main()