#!/usr/bin/env python3
"""
Sacramento Model Verification Benchmark

Main script for generating test data, running the Python Sacramento model,
and comparing outputs against C# reference (when available).

Usage:
    python run_benchmark.py --generate    # Generate synthetic data
    python run_benchmark.py --verify      # Compare C# vs Python outputs
    python run_benchmark.py --report      # Generate verification report
    python run_benchmark.py --all         # Run all steps
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.generate_test_data import generate_all_test_data
from benchmark.verify_implementation import run_all_verifications, save_python_outputs


def generate_verification_report(results: dict, output_path: Path) -> None:
    """
    Generate a markdown verification report.
    
    Args:
        results: Dictionary of test results
        output_path: Path to write report
    """
    report_lines = []
    
    report_lines.append("# Sacramento Model Verification Report")
    report_lines.append("")
    report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Python Version**: {sys.version.split()[0]}")
    report_lines.append("")
    
    # Summary table
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Test Case | Status | Max Difference | Variable |")
    report_lines.append("|-----------|--------|----------------|----------|")
    
    all_pass = True
    for test_id, result in results.items():
        status = "PASS" if result['comparison']['pass'] else "FAIL"
        if not result['comparison']['pass']:
            all_pass = False
        
        # Find the variable with max difference
        max_diff = 0.0
        max_var = "N/A"
        if result['comparison'].get('summary'):
            for var, stats in result['comparison']['summary'].items():
                if stats['max_difference'] > max_diff:
                    max_diff = stats['max_difference']
                    max_var = var
        
        report_lines.append(
            f"| {result['test_name']} | {status} | {max_diff:.2e} | {max_var} |"
        )
    
    report_lines.append("")
    report_lines.append("## Overall Result")
    report_lines.append("")
    if all_pass:
        report_lines.append("**ALL TESTS PASSED**")
    else:
        report_lines.append("**SOME TESTS FAILED** - See details below")
    
    # Detailed results
    report_lines.append("")
    report_lines.append("## Detailed Results")
    report_lines.append("")
    
    for test_id, result in results.items():
        report_lines.append(f"### {result['test_name']}")
        report_lines.append("")
        report_lines.append(f"- Input records: {result['input_records']}")
        report_lines.append(f"- C# reference available: {result['has_csharp_reference']}")
        report_lines.append(f"- Status: {'PASS' if result['comparison']['pass'] else 'FAIL'}")
        report_lines.append("")
        
        if result['comparison'].get('summary'):
            report_lines.append("**Variable Statistics:**")
            report_lines.append("")
            report_lines.append("| Variable | Max Difference | Mean Difference | Timestep of Max |")
            report_lines.append("|----------|----------------|-----------------|-----------------|")
            
            for var, stats in result['comparison']['summary'].items():
                report_lines.append(
                    f"| {var} | {stats['max_difference']:.2e} | "
                    f"{stats['mean_difference']:.2e} | {stats['timestep_of_max']} |"
                )
            report_lines.append("")
        
        if result['comparison'].get('differences'):
            report_lines.append("**Failures:**")
            report_lines.append("")
            for diff in result['comparison']['differences']:
                if 'max_difference' in diff:
                    report_lines.append(
                        f"- {diff['variable']}: max difference {diff['max_difference']:.2e} "
                        f"at timestep {diff['timestep_of_max']} (tolerance: {diff['tolerance']:.2e})"
                    )
                elif 'issue' in diff:
                    report_lines.append(f"- {diff['variable']}: {diff['issue']}")
            report_lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Verification report saved to: {output_path}")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Sacramento Model Verification Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmark.py --generate    # Generate synthetic test data
    python run_benchmark.py --verify      # Run verification tests
    python run_benchmark.py --report      # Generate report (after verify)
    python run_benchmark.py --all         # Run all steps
        """
    )
    parser.add_argument('--generate', action='store_true',
                        help='Generate synthetic test data')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification tests')
    parser.add_argument('--report', action='store_true',
                        help='Generate verification report')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps (generate, verify, report)')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                        help='Tolerance for comparisons (default: 1e-10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for test data')
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not any([args.generate, args.verify, args.report, args.all]):
        parser.print_help()
        return 0
    
    # Determine directories
    script_dir = Path(__file__).parent.parent
    test_data_dir = Path(args.output_dir) if args.output_dir else script_dir / "test_data"
    
    results = None
    
    # Generate test data
    if args.generate or args.all:
        print("\n" + "=" * 60)
        print("Step 1: Generating Test Data")
        print("=" * 60)
        generate_all_test_data(test_data_dir)
    
    # Run verification
    if args.verify or args.all:
        print("\n" + "=" * 60)
        print("Step 2: Running Verification Tests")
        print("=" * 60)
        results = run_all_verifications(test_data_dir, tolerance=args.tolerance)
        save_python_outputs(results, test_data_dir)
    
    # Generate report
    if args.report or args.all:
        print("\n" + "=" * 60)
        print("Step 3: Generating Verification Report")
        print("=" * 60)
        
        if results is None:
            # Need to run verification first
            print("Running verification to generate report...")
            results = run_all_verifications(test_data_dir, tolerance=args.tolerance)
        
        report_path = script_dir / "verification_report.md"
        generate_verification_report(results, report_path)
    
    # Final summary
    if results is not None:
        all_pass = all(r['comparison']['pass'] for r in results.values())
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        
        if all_pass:
            print("SUCCESS: All verification tests passed!")
            return 0
        else:
            print("FAILURE: Some verification tests failed!")
            for test_id, result in results.items():
                if not result['comparison']['pass']:
                    print(f"  - {result['test_name']}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
