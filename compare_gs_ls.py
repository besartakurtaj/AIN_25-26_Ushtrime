import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from parser.parser import Parser
from scheduler.genetic_algorithm import GeneticScheduler
from scheduler.local_search_scheduler import LocalSearchScheduler


def load_instance(instance_path: str):
    """Load an instance from JSON file."""
    parser = Parser()
    return parser.parse(instance_path)


def run_ga_optimization(instance_data, num_runs: int = 10, params: Dict = None, 
                        time_limit: int = 60, verbose: bool = False) -> Dict:
    """
    Run GA optimization multiple times and collect statistics.
    """
    if params is None:
        params = {
            "POP_SIZE": 15,
            "GENERATIONS": 25,
            "CROSSOVER_RATE": 0.80,
            "MUTATION_RATE": 0.20,
            "TOURNAMENT_SIZE": 2,
            "ELITISM": 1,
            "TIME_LIMIT": time_limit,
        }
    
    runs = []
    
    for run_num in range(1, num_runs + 1):
        start = time.time()
        
        scheduler = GeneticScheduler(
            instance_data=instance_data,
            verbose=verbose,
            params=params,
            initial_solution=None,
        )
        
        solution = scheduler.generate_solution(start_time=start)
        elapsed = time.time() - start
        
        runs.append({
            "run": run_num,
            "score": solution.total_score,
            "time_seconds": elapsed,
            "num_scheduled": len(solution.scheduled_programs),
        })
        
        if verbose:
            print(f"  GA Run {run_num}/{num_runs}: score={solution.total_score}, time={elapsed:.2f}s")
    
    # Calculate statistics
    scores = [r["score"] for r in runs]
    times = [r["time_seconds"] for r in runs]
    
    stats = {
        "best": max(scores),
        "worst": min(scores),
        "average": sum(scores) / len(scores),
        "std_dev": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
    }
    
    return {
        "algorithm": "Genetic Algorithm",
        "parameters": params,
        "runs": runs,
        "statistics": stats,
    }


def run_ls_optimization(instance_data, num_runs: int = 10, params: Dict = None,
                        time_limit: int = 60, verbose: bool = False) -> Dict:
    """
    Run Local Search optimization multiple times and collect statistics.
    """
    if params is None:
        params = {
            "MAX_ITERATIONS": 500,
            "RESTART_ITERATIONS": 3,
            "MAX_NO_IMPROVE": 100,
            "TIME_LIMIT": time_limit,
            "FIRST_IMPROVEMENT": True,
        }
    
    runs = []
    
    for run_num in range(1, num_runs + 1):
        start = time.time()
        
        scheduler = LocalSearchScheduler(
            instance_data=instance_data,
            verbose=verbose,
            params=params,
            initial_solution=None,
        )
        
        solution = scheduler.generate_solution(start_time=start)
        elapsed = time.time() - start
        
        runs.append({
            "run": run_num,
            "score": solution.total_score,
            "time_seconds": elapsed,
            "num_scheduled": len(solution.scheduled_programs),
        })
        
        if verbose:
            print(f"  LS Run {run_num}/{num_runs}: score={solution.total_score}, time={elapsed:.2f}s")
    
    # Calculate statistics
    scores = [r["score"] for r in runs]
    times = [r["time_seconds"] for r in runs]
    
    stats = {
        "best": max(scores),
        "worst": min(scores),
        "average": sum(scores) / len(scores),
        "std_dev": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
    }
    
    return {
        "algorithm": "Local Search",
        "parameters": params,
        "runs": runs,
        "statistics": stats,
    }


def run_hybrid_ga_ls(instance_data, num_runs: int = 10, 
                     ga_params: Dict = None, ls_params: Dict = None,
                     time_limit: int = 60, verbose: bool = False) -> Dict:
    """
    Run Genetic Algorithm first, then apply Local Search to the result.
    This shows the improvement when GA is refined with LS.
    """
    if ga_params is None:
        ga_params = {
            "POP_SIZE": 15,
            "GENERATIONS": 25,
            "CROSSOVER_RATE": 0.80,
            "MUTATION_RATE": 0.20,
            "TOURNAMENT_SIZE": 2,
            "ELITISM": 1,
            "TIME_LIMIT": time_limit // 2,
        }
    
    if ls_params is None:
        ls_params = {
            "MAX_ITERATIONS": 300,
            "RESTART_ITERATIONS": 1,
            "MAX_NO_IMPROVE": 50,
            "TIME_LIMIT": time_limit // 2,
            "FIRST_IMPROVEMENT": True,
        }
    
    runs = []
    
    for run_num in range(1, num_runs + 1):
        run_start = time.time()
        
        # Run GA first
        ga_start = time.time()
        ga_scheduler = GeneticScheduler(
            instance_data=instance_data,
            verbose=False,
            params=ga_params,
            initial_solution=None,
        )
        ga_solution = ga_scheduler.generate_solution(start_time=ga_start)
        ga_time = time.time() - ga_start
        ga_score = ga_solution.total_score
        
        # Apply LS to GA result
        ls_start = time.time()
        ls_scheduler = LocalSearchScheduler(
            instance_data=instance_data,
            verbose=False,
            params=ls_params,
            initial_solution=ga_solution,
        )
        final_solution = ls_scheduler.generate_solution(start_time=ls_start)
        ls_time = time.time() - ls_start
        final_score = final_solution.total_score
        
        elapsed = time.time() - run_start
        improvement = final_score - ga_score
        improvement_pct = (improvement / ga_score * 100) if ga_score > 0 else 0
        
        runs.append({
            "run": run_num,
            "ga_score": ga_score,
            "ls_score": final_score,
            "ga_time": ga_time,
            "ls_time": ls_time,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "total_time": elapsed,
            "num_scheduled": len(final_solution.scheduled_programs),
        })
        
        if verbose:
            print(f"  Hybrid Run {run_num}/{num_runs}: "
                  f"GA={ga_score} -> LS={final_score} "
                  f"(+{improvement} / {improvement_pct:.2f}%), "
                  f"time={elapsed:.2f}s")
    
    # Calculate statistics
    ga_scores = [r["ga_score"] for r in runs]
    ls_scores = [r["ls_score"] for r in runs]
    improvements = [r["improvement"] for r in runs]
    improvement_pcts = [r["improvement_pct"] for r in runs]
    
    avg_improvement = sum(improvements) / len(improvements)
    avg_improvement_pct = sum(improvement_pcts) / len(improvement_pcts)
    
    stats = {
        "ga_best": max(ga_scores),
        "ga_average": sum(ga_scores) / len(ga_scores),
        "ls_best": max(ls_scores),
        "ls_average": sum(ls_scores) / len(ls_scores),
        "improvement_avg": avg_improvement,
        "improvement_avg_pct": avg_improvement_pct,
        "improvement_best": max(improvements),
        "improvement_worst": min(improvements),
        "avg_time": sum(r["total_time"] for r in runs) / len(runs),
        "total_time": sum(r["total_time"] for r in runs),
    }
    
    return {
        "algorithm": "GA + Local Search",
        "ga_parameters": ga_params,
        "ls_parameters": ls_params,
        "runs": runs,
        "statistics": stats,
    }


def compare_instances(instances_dir: str, output_dir: str, 
                      num_runs: int = 10, time_limit: int = 60, verbose: bool = False):
    """
    Run comparison on all instances in a directory.
    """
    instances_path = Path(instances_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    instance_files = sorted(instances_path.glob("*.json"))
    
    if not instance_files:
        print(f"No instance files found in {instances_dir}")
        return
    
    print(f"Found {len(instance_files)} instances")
    print(f"Running {num_runs} executions per algorithm per instance")
    print(f"Time limit per algorithm: {time_limit} seconds")
    print("")
    
    all_results = []
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_instances": len(instance_files),
        "num_runs_per_instance": num_runs,
        "time_limit_seconds": time_limit,
        "instances": [],
    }
    
    for i, instance_file in enumerate(instance_files, 1):
        instance_name = instance_file.stem
        print(f"[{i}/{len(instance_files)}] Processing {instance_name}...")
        
        try:
            instance_data = load_instance(str(instance_file))
            print(f"  Instance loaded: {len(instance_data.channels)} channels, "
                  f"{sum(len(ch.programs) for ch in instance_data.channels)} programs")
            
            # Run GA
            print(f"  Running Genetic Algorithm ({num_runs} executions)...")
            ga_results = run_ga_optimization(instance_data, num_runs, time_limit=time_limit, verbose=verbose)
            
            # Run LS
            print(f"  Running Local Search ({num_runs} executions)...")
            ls_results = run_ls_optimization(instance_data, num_runs, time_limit=time_limit, verbose=verbose)
            
            # Run Hybrid GA+LS
            print(f"  Running Hybrid GA+LS ({num_runs} executions)...")
            hybrid_results = run_hybrid_ga_ls(instance_data, num_runs, 
                                             time_limit=time_limit, verbose=verbose)
            
            # Prepare comparison
            comparison = {
                "instance": instance_name,
                "ga": ga_results,
                "ls": ls_results,
                "hybrid": hybrid_results,
                "comparison": {
                    "ga_vs_ls_avg": ls_results["statistics"]["average"] - ga_results["statistics"]["average"],
                    "ga_vs_ls_best": ls_results["statistics"]["best"] - ga_results["statistics"]["best"],
                    "hybrid_vs_ga_avg": hybrid_results["statistics"]["ls_average"] - hybrid_results["statistics"]["ga_average"],
                    "hybrid_vs_ga_best": hybrid_results["statistics"]["ls_best"] - hybrid_results["statistics"]["ga_best"],
                },
            }
            
            all_results.append(comparison)
            summary["instances"].append({
                "name": instance_name,
                "ga_avg_score": ga_results["statistics"]["average"],
                "ls_avg_score": ls_results["statistics"]["average"],
                "hybrid_avg_score": hybrid_results["statistics"]["ls_average"],
                "ga_vs_ls_improvement": comparison["comparison"]["ga_vs_ls_avg"],
                "hybrid_vs_ga_improvement": comparison["comparison"]["hybrid_vs_ga_avg"],
            })
            
            print(f"  GA avg: {ga_results['statistics']['average']:.1f}")
            print(f"  LS avg: {ls_results['statistics']['average']:.1f} "
                  f"(+{comparison['comparison']['ga_vs_ls_avg']:.1f})")
            print(f"  Hybrid avg: {hybrid_results['statistics']['ls_average']:.1f} "
                  f"(+{comparison['comparison']['hybrid_vs_ga_avg']:.1f})")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
    
    # Save detailed results
    results_file = output_path / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "detailed_results": all_results,
        }, f, indent=2)
    print(f"Detailed results saved to {results_file}")
    
    # Generate and save summary report
    report_file = output_path / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report = generate_comparison_report(summary, all_results)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Summary report saved to {report_file}")
    
    print("\n" + report)


def generate_comparison_report(summary: Dict, all_results: List) -> str:
    """Generate a comparison report."""
    report = []
    report.append("=" * 100)
    report.append("GA vs LOCAL SEARCH vs HYBRID COMPARISON REPORT")
    report.append("=" * 100)
    report.append(f"Date: {summary['timestamp']}")
    report.append(f"Instances: {summary['num_instances']}")
    report.append(f"Runs per instance: {summary['num_runs_per_instance']}")
    report.append(f"Time limit per algorithm: {summary['time_limit_seconds']}s")
    report.append("")
    
    report.append("-" * 100)
    report.append("SUMMARY BY INSTANCE")
    report.append("-" * 100)
    
    ga_total_avg = 0
    ls_total_avg = 0
    hybrid_total_avg = 0
    
    for inst in summary["instances"]:
        ga_total_avg += inst["ga_avg_score"]
        ls_total_avg += inst["ls_avg_score"]
        hybrid_total_avg += inst["hybrid_avg_score"]
        
        report.append(f"\n{inst['name']}")
        report.append(f"  GA:     {inst['ga_avg_score']:.1f}")
        report.append(f"  LS:     {inst['ls_avg_score']:.1f}  " 
                      f"({inst['ga_vs_ls_improvement']:+.1f} vs GA)")
        report.append(f"  Hybrid: {inst['hybrid_avg_score']:.1f}  "
                      f"({inst['hybrid_vs_ga_improvement']:+.1f} vs GA)")
    
    n = len(summary["instances"])
    report.append("\n" + "-" * 100)
    report.append("OVERALL AVERAGES")
    report.append("-" * 100)
    report.append(f"GA avg:     {ga_total_avg/n:.1f}")
    report.append(f"LS avg:     {ls_total_avg/n:.1f}  ({(ls_total_avg-ga_total_avg)/n:+.1f})")
    report.append(f"Hybrid avg: {hybrid_total_avg/n:.1f}  ({(hybrid_total_avg-ga_total_avg)/n:+.1f})")
    report.append("")
    
    # Statistics
    ls_improvements = [inst["ga_vs_ls_improvement"] for inst in summary["instances"]]
    hybrid_improvements = [inst["hybrid_vs_ga_improvement"] for inst in summary["instances"]]
    
    avg_ls_imp = sum(ls_improvements) / len(ls_improvements)
    avg_hybrid_imp = sum(hybrid_improvements) / len(hybrid_improvements)
    
    report.append("LS improvements over GA:")
    report.append(f"  Average: +{avg_ls_imp:.1f} ({avg_ls_imp/(ga_total_avg/n)*100:.2f}%)")
    report.append(f"  Best: +{max(ls_improvements):.1f}")
    report.append(f"  Worst: {min(ls_improvements):+.1f}")
    report.append(f"  Positive improvements: {sum(1 for x in ls_improvements if x > 0)}/{len(ls_improvements)}")
    report.append("")
    
    report.append("Hybrid (GA+LS) improvements over GA alone:")
    report.append(f"  Average: +{avg_hybrid_imp:.1f} ({avg_hybrid_imp/(ga_total_avg/n)*100:.2f}%)")
    report.append(f"  Best: +{max(hybrid_improvements):.1f}")
    report.append(f"  Worst: {min(hybrid_improvements):+.1f}")
    report.append(f"  Positive improvements: {sum(1 for x in hybrid_improvements if x > 0)}/{len(hybrid_improvements)}")
    
    report.append("")
    report.append("=" * 100)
    
    return "\n".join(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA vs LS comparison")
    parser.add_argument("--instances", default="data/input",
                        help="Directory with instance files")
    parser.add_argument("--output", default="results/comparison",
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per algorithm per instance")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Time limit per algorithm in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    compare_instances(
        args.instances,
        args.output,
        num_runs=args.runs,
        time_limit=args.time_limit,
        verbose=args.verbose
    )
