"""
COMPREHENSIVE OPTIMIZATION AND ANALYSIS RUNNER
===============================================

Main script that runs:
1. Parameter Analysis - analyze optimal GA parameters from existing results
2. Per-Instance Analysis - detailed results for each instance separately
3. Comparison - run 10 executions of GA, LS, and Hybrid GA+LS on all instances
4. Report Generation - generate comprehensive comparison reports
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from parameter_analysis import generate_analysis_report
from per_instance_analysis import run_all_instances
from compare_ga_ls import compare_instances


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive optimization and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python run_optimization.py
  
  # Run with custom settings
  python run_optimization.py --runs 15 --time-limit 120 --verbose
  
  # Skip parameter analysis
  python run_optimization.py --skip-analysis
        """
    )
    
    parser.add_argument("--instances", default="data/input",
                        help="Directory with instance files (default: data/input)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory with existing GA results (default: results)")
    parser.add_argument("--output", default="results/optimization_run",
                        help="Output directory for comparison results")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per algorithm per instance (default: 10)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Time limit per algorithm in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip parameter analysis phase")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip comparison phase")
    parser.add_argument("--skip-per-instance", action="store_true",
                        help="Skip per-instance detailed analysis phase")
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("COMPREHENSIVE OPTIMIZATION AND ANALYSIS")
    print("=" * 100)
    print("")
    
    # Phase 1: Parameter Analysis
    if not args.skip_analysis:
        print("PHASE 1: PARAMETER ANALYSIS")
        print("-" * 100)
        print("Analyzing existing GA results to identify optimal parameters...")
        print("")
        
        try:
            report = generate_analysis_report(args.results_dir)
            print(report)
            
            # Save analysis report
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            analysis_report_file = output_path / f"parameter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(analysis_report_file, "w") as f:
                f.write(report)
            print(f"\nParameter analysis report saved to {analysis_report_file}")
            
        except Exception as e:
            print(f"ERROR during parameter analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        print("")
        print("=" * 100)
        print("")
    
    # Phase 2: Per-Instance Analysis
    if not args.skip_per_instance:
        print("PHASE 2: PER-INSTANCE OPTIMIZATION ANALYSIS")
        print("-" * 100)
        print(f"Running detailed analysis for each instance separately")
        print(f"Instances: {args.instances}")
        print(f"Runs per instance: {args.runs}")
        print(f"Time limit per algorithm: {args.time_limit}s")
        print("")
        
        try:
            run_all_instances(
                instances_dir=args.instances,
                output_dir=args.output,
                num_runs=args.runs,
                ga_time_limit=args.time_limit,
                ls_time_limit=args.time_limit,
                verbose=args.verbose
            )
            
        except Exception as e:
            print(f"ERROR during per-instance analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        print("")
        print("=" * 100)
        print("")
    
    # Phase 3: Comparison (GA vs LS vs Hybrid)
    if not args.skip_comparison:
        print("PHASE 3: GA vs LOCAL SEARCH COMPARISON")
        print("-" * 100)
        print(f"Comparing Genetic Algorithm, Local Search, and Hybrid approaches")
        print(f"Instances: {args.instances}")
        print(f"Runs per instance: {args.runs}")
        print(f"Time limit per algorithm: {args.time_limit}s")
        print("")
        
        try:
            compare_instances(
                instances_dir=args.instances,
                output_dir=args.output,
                num_runs=args.runs,
                time_limit=args.time_limit,
                verbose=args.verbose
            )
            
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print("")
    print("=" * 100)
    print("OPTIMIZATION AND ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
