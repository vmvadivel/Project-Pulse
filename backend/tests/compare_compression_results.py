"""
Compare compression test results and generate report.
Usage: python tests/compare_compression_results.py
"""

import json
import sys
from pathlib import Path

class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'


def format_size(kb: float) -> str:
    """Format KB to human readable"""
    if kb < 1:
        return f"{kb * 1024:.0f} B"
    elif kb < 1024:
        return f"{kb:.2f} KB"
    else:
        return f"{kb / 1024:.2f} MB"


def calculate_improvement(before: float, after: float) -> tuple:
    """Calculate improvement percentage and return formatted string"""
    if before == 0:
        return 0, "N/A"
    
    improvement = ((before - after) / before) * 100
    if improvement > 0:
        return improvement, f"{Colors.GREEN}↓ {improvement:.1f}%{Colors.NC}"
    elif improvement < 0:
        return improvement, f"{Colors.RED}↑ {abs(improvement):.1f}%{Colors.NC}"
    else:
        return 0, "No change"


def load_results(filename: str) -> dict:
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def compare_endpoints(before_data: dict, after_data: dict):
    """Compare endpoint-by-endpoint performance"""
    
    print(f"\n{Colors.BOLD}ENDPOINT COMPARISON{Colors.NC}")
    print("="*100)
    
    # Create mapping of endpoints
    before_endpoints = {ep['endpoint']: ep for ep in before_data.get('endpoints', [])}
    after_endpoints = {ep['endpoint']: ep for ep in after_data.get('endpoints', [])}
    
    # Header
    print(f"\n{'Endpoint':<20} {'Before (KB)':<15} {'After (KB)':<15} {'Saved':<15} {'Latency Change':<20}")
    print("-"*100)
    
    for endpoint_name in sorted(before_endpoints.keys()):
        if endpoint_name not in after_endpoints:
            continue
        
        before = before_endpoints[endpoint_name]
        after = after_endpoints[endpoint_name]
        
        # Size comparison
        before_size = before.get('avg_size_kb', 0)
        after_size = after.get('avg_size_compressed_kb', after.get('avg_size_kb', 0))
        size_saved_pct, size_str = calculate_improvement(before_size, after_size)
        
        # Latency comparison
        before_latency = before.get('avg_latency_ms', 0)
        after_latency = after.get('avg_latency_ms', 0)
        latency_improvement, latency_str = calculate_improvement(before_latency, after_latency)
        
        print(
            f"{endpoint_name:<20} "
            f"{before_size:>9.2f}      "
            f"{after_size:>9.2f}      "
            f"{size_str:<23}  "
            f"{latency_str:<28}"
        )


def generate_summary(before_data: dict, after_data: dict):
    """Generate overall summary"""
    
    print(f"\n{Colors.BOLD}OVERALL SUMMARY{Colors.NC}")
    print("="*100)
    
    # Data transfer comparison
    before_total_kb = before_data.get('total_data_kb', 0)
    after_total_kb = after_data.get('total_data_compressed_kb', 
                                     after_data.get('total_data_kb', 0))
    
    data_saved_kb = before_total_kb - after_total_kb
    data_saved_pct = (data_saved_kb / before_total_kb * 100) if before_total_kb > 0 else 0
    
    print(f"\n{Colors.CYAN}📊 Data Transfer:{Colors.NC}")
    print(f"  Before Compression: {format_size(before_total_kb)}")
    print(f"  After Compression:  {format_size(after_total_kb)}")
    print(f"  {Colors.GREEN}Bandwidth Saved:    {format_size(data_saved_kb)} ({data_saved_pct:.1f}% reduction){Colors.NC}")
    
    # Response size comparison
    before_avg_kb = before_data.get('avg_response_size_kb', 0)
    after_avg_kb = after_data.get('avg_response_size_compressed_kb', 
                                   after_data.get('avg_response_size_kb', 0))
    
    print(f"\n{Colors.CYAN}📦 Average Response Size:{Colors.NC}")
    print(f"  Before: {format_size(before_avg_kb)}")
    print(f"  After:  {format_size(after_avg_kb)}")
    print(f"  Change: {Colors.GREEN}↓ {((before_avg_kb - after_avg_kb) / before_avg_kb * 100):.1f}%{Colors.NC}")
    
    # Latency comparison
    before_latency = before_data.get('avg_latency_ms', 0)
    after_latency = after_data.get('avg_latency_ms', 0)
    latency_change = before_latency - after_latency
    latency_pct = (latency_change / before_latency * 100) if before_latency > 0 else 0
    
    print(f"\n{Colors.CYAN}⚡ Latency:{Colors.NC}")
    print(f"  Before: {before_latency:.0f} ms")
    print(f"  After:  {after_latency:.0f} ms")
    if latency_change > 0:
        print(f"  Change: {Colors.GREEN}↓ {latency_change:.0f} ms ({latency_pct:.1f}% faster){Colors.NC}")
    elif latency_change < 0:
        print(f"  Change: {Colors.RED}↑ {abs(latency_change):.0f} ms ({abs(latency_pct):.1f}% slower){Colors.NC}")
    else:
        print(f"  Change: No significant change")
    
    # Compression stats
    if 'compressed_requests' in after_data:
        comp_rate = after_data.get('compression_rate', 0)
        print(f"\n{Colors.CYAN}🗜️  Compression:{Colors.NC}")
        print(f"  Requests Compressed: {after_data['compressed_requests']}/{after_data['total_requests']} ({comp_rate:.1f}%)")
        print(f"  Bandwidth Saved: {after_data.get('bandwidth_saved_kb', 0):.2f} KB ({after_data.get('bandwidth_saved_percent', 0):.1f}%)")


def calculate_bandwidth_cost_savings(data_saved_kb: float, requests_per_month: int = 1_000_000):
    """Estimate cost savings from bandwidth reduction"""
    
    # Typical bandwidth costs (AWS CloudFront example)
    # First 10TB: $0.085/GB
    # Next 40TB: $0.080/GB
    # Over 150TB: $0.060/GB
    
    # Use average of $0.075/GB
    cost_per_gb = 0.075
    
    # Calculate monthly savings
    data_saved_gb = data_saved_kb / (1024 * 1024)
    savings_per_request = data_saved_gb * cost_per_gb
    monthly_savings = savings_per_request * requests_per_month
    yearly_savings = monthly_savings * 12
    
    return monthly_savings, yearly_savings


def generate_cost_estimate(before_data: dict, after_data: dict):
    """Generate cost savings estimate"""
    
    print(f"\n{Colors.BOLD}COST SAVINGS ESTIMATE{Colors.NC}")
    print("="*100)
    
    before_avg_kb = before_data.get('avg_response_size_kb', 0)
    after_avg_kb = after_data.get('avg_response_size_compressed_kb', 
                                   after_data.get('avg_response_size_kb', 0))
    saved_kb = before_avg_kb - after_avg_kb
    
    if saved_kb <= 0:
        print("\nNo bandwidth savings to calculate.")
        return
    
    print(f"\n{Colors.CYAN}Assumptions:{Colors.NC}")
    print(f"  - Bandwidth saved per request: {format_size(saved_kb)}")
    print(f"  - Bandwidth cost: $0.075/GB (AWS CloudFront average)")
    
    # Calculate for different traffic levels
    traffic_levels = [
        (10_000, "10K requests/month (Low traffic)"),
        (100_000, "100K requests/month (Medium traffic)"),
        (1_000_000, "1M requests/month (High traffic)"),
        (10_000_000, "10M requests/month (Very high traffic)")
    ]
    
    print(f"\n{'Traffic Level':<40} {'Monthly Savings':<20} {'Yearly Savings':<20}")
    print("-"*80)
    
    for requests, label in traffic_levels:
        monthly, yearly = calculate_bandwidth_cost_savings(saved_kb, requests)
        print(f"{label:<40} ${monthly:>18.2f}  ${yearly:>18.2f}")
    
    # Additional benefits
    print(f"\n{Colors.CYAN}Additional Benefits:{Colors.NC}")
    print("  ✓ Reduced bandwidth costs")
    print("  ✓ Faster page load times")
    print("  ✓ Improved user experience")
    print("  ✓ Better mobile performance")
    print("  ✓ Lower CDN costs")
    print("  ✓ Reduced server egress charges")


def generate_recommendation(before_data: dict, after_data: dict):
    """Generate recommendation based on results"""
    
    print(f"\n{Colors.BOLD}RECOMMENDATION{Colors.NC}")
    print("="*100)
    
    before_avg_kb = before_data.get('avg_response_size_kb', 0)
    after_avg_kb = after_data.get('avg_response_size_compressed_kb', 
                                   after_data.get('avg_response_size_kb', 0))
    
    bandwidth_saved_pct = ((before_avg_kb - after_avg_kb) / before_avg_kb * 100) if before_avg_kb > 0 else 0
    
    before_latency = before_data.get('avg_latency_ms', 0)
    after_latency = after_data.get('avg_latency_ms', 0)
    latency_change_pct = ((before_latency - after_latency) / before_latency * 100) if before_latency > 0 else 0
    
    print()
    
    # Decision logic
    if bandwidth_saved_pct > 40 and latency_change_pct >= -5:
        print(f"{Colors.GREEN}✅ HIGHLY RECOMMENDED{Colors.NC}")
        print(f"\nCompression provides excellent benefits:")
        print(f"  • {bandwidth_saved_pct:.1f}% bandwidth reduction")
        if latency_change_pct > 0:
            print(f"  • {latency_change_pct:.1f}% latency improvement")
        else:
            print(f"  • Minimal latency impact ({abs(latency_change_pct):.1f}%)")
        print(f"\n{Colors.BOLD}Action:{Colors.NC} Deploy compression to production immediately.")
        
    elif bandwidth_saved_pct > 20 and latency_change_pct >= -10:
        print(f"{Colors.GREEN}✅ RECOMMENDED{Colors.NC}")
        print(f"\nCompression provides good benefits:")
        print(f"  • {bandwidth_saved_pct:.1f}% bandwidth reduction")
        if latency_change_pct > 0:
            print(f"  • {latency_change_pct:.1f}% latency improvement")
        else:
            print(f"  • Acceptable latency impact ({abs(latency_change_pct):.1f}%)")
        print(f"\n{Colors.BOLD}Action:{Colors.NC} Deploy compression to production.")
        
    elif latency_change_pct < -20:
        print(f"{Colors.YELLOW}⚠️  USE WITH CAUTION{Colors.NC}")
        print(f"\nCompression increases latency significantly:")
        print(f"  • {abs(latency_change_pct):.1f}% latency increase")
        print(f"  • {bandwidth_saved_pct:.1f}% bandwidth savings")
        print(f"\n{Colors.BOLD}Action:{Colors.NC} Consider:")
        print("  1. Increasing compression level threshold (minimum_size)")
        print("  2. Reducing compression level (1-5 instead of 6-9)")
        print("  3. Using compression only for large responses")
        
    else:
        print(f"{Colors.YELLOW}⚠️  LIMITED BENEFIT{Colors.NC}")
        print(f"\nCompression provides minimal benefits:")
        print(f"  • {bandwidth_saved_pct:.1f}% bandwidth reduction")
        print(f"  • {latency_change_pct:.1f}% latency change")
        print(f"\n{Colors.BOLD}Action:{Colors.NC} Consider if bandwidth costs justify the complexity.")
    
    # Best practices
    print(f"\n{Colors.CYAN}Best Practices:{Colors.NC}")
    print("  1. Set minimum size threshold (500-1000 bytes)")
    print("  2. Use compression level 6 (balanced) or 4-5 (faster)")
    print("  3. Compress JSON, HTML, XML, CSS, and JavaScript")
    print("  4. Don't compress images, videos, or already-compressed files")
    print("  5. Monitor compression middleware statistics")
    print("  6. Add Vary: Accept-Encoding header for caching")


def main():
    print("="*100)
    print(f"{Colors.BOLD}GZIP COMPRESSION - BEFORE/AFTER COMPARISON{Colors.NC}")
    print("="*100)
    
    # Load results
    before_data = load_results('results_before_compression.json')
    after_data = load_results('results_after_compression.json')
    
    if not before_data:
        print(f"\n{Colors.RED}ERROR: Could not load results_before_compression.json{Colors.NC}")
        print("Please run: python test_before_compression.py")
        sys.exit(1)
    
    if not after_data:
        print(f"\n{Colors.RED}ERROR: Could not load results_after_compression.json{Colors.NC}")
        print("Please run: python test_after_compression.py")
        sys.exit(1)
    
    print(f"\n{Colors.GREEN}✓ Loaded test results{Colors.NC}")
    print(f"  Before: {before_data.get('timestamp', 'Unknown time')}")
    print(f"  After:  {after_data.get('timestamp', 'Unknown time')}")
    
    # Generate reports
    generate_summary(before_data, after_data)
    compare_endpoints(before_data, after_data)
    generate_cost_estimate(before_data, after_data)
    generate_recommendation(before_data, after_data)
    
    # Save comparison report
    comparison = {
        'test_dates': {
            'before': before_data.get('timestamp'),
            'after': after_data.get('timestamp')
        },
        'bandwidth_savings': {
            'before_total_kb': before_data.get('total_data_kb', 0),
            'after_total_kb': after_data.get('total_data_compressed_kb', 0),
            'saved_kb': before_data.get('total_data_kb', 0) - after_data.get('total_data_compressed_kb', 0),
            'saved_percent': ((before_data.get('total_data_kb', 0) - after_data.get('total_data_compressed_kb', 0)) / 
                             before_data.get('total_data_kb', 1)) * 100
        },
        'latency_comparison': {
            'before_ms': before_data.get('avg_latency_ms', 0),
            'after_ms': after_data.get('avg_latency_ms', 0),
            'change_ms': before_data.get('avg_latency_ms', 0) - after_data.get('avg_latency_ms', 0),
            'change_percent': ((before_data.get('avg_latency_ms', 0) - after_data.get('avg_latency_ms', 0)) / 
                              before_data.get('avg_latency_ms', 1)) * 100
        },
        'compression_stats': {
            'compression_rate': after_data.get('compression_rate', 0),
            'compressed_requests': after_data.get('compressed_requests', 0),
            'total_requests': after_data.get('total_requests', 0)
        }
    }
    
    with open('compression_comparison_report.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{Colors.GREEN}✓ Comparison report saved to: compression_comparison_report.json{Colors.NC}")
    print("\n" + "="*100)


if __name__ == "__main__":
    main()