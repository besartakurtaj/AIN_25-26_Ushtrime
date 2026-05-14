import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from parser.parser import Parser
from scheduler.genetic_algorithm import GeneticScheduler
from scheduler.local_search_scheduler import LocalSearchScheduler


def load_instance(instance_path: str):
    """Load an instance from JSON file."""
    parser = Parser(instance_path)
    return parser.parse()


def run_instance_analysis(instance_name: str, instance_path: str, num_runs: int = 10,
                          ga_time_limit: int = 60, ls_time_limit: int = 60,
                          verbose: bool = False) -> Dict:
    """
    Run complete analysis for a single instance:
    - GA (10 runs)
    - LS (10 runs) with improvement calculation
    - Hybrid GA+LS (10 runs) with improvement calculation
    """
    print(f"\n{'='*100}")
    print(f"INSTANCE: {instance_name}")
    print(f"{'='*100}")
    
    # Load instance
    try:
        instance_data = load_instance(instance_path)
        num_channels = len(instance_data.channels)
        num_programs = sum(len(ch.programs) for ch in instance_data.channels)
        print(f"Channels: {num_channels}, Programs: {num_programs}")
    except Exception as e:
        print(f"ERROR loading instance: {e}")
        return None
    
    results = {
        "instance": instance_name,
        "instance_path": instance_path,
        "timestamp": datetime.now().isoformat(),
        "instance_info": {
            "num_channels": num_channels,
            "num_programs": num_programs,
        },
    }
    
    # ========== RUN GA ==========
    print(f"\n[GA] Running Genetic Algorithm ({num_runs} executions)...")
    ga_runs = []
    ga_start_time = time.time()
    
    for run_num in range(1, num_runs + 1):
        run_start = time.time()
        scheduler = GeneticScheduler(
            instance_data=instance_data,
            verbose=False,
            params={
                "POP_SIZE": 15,
                "GENERATIONS": 25,
                "CROSSOVER_RATE": 0.80,
                "MUTATION_RATE": 0.20,
                "TOURNAMENT_SIZE": 2,
                "ELITISM": 1,
                "TIME_LIMIT": ga_time_limit,
            },
        )
        solution = scheduler.generate_solution(start_time=run_start)
        elapsed = time.time() - run_start
        
        ga_runs.append({
            "run": run_num,
            "score": solution.total_score,
            "time_seconds": elapsed,
            "programs_scheduled": len(solution.scheduled_programs),
        })
        print(f"  Run {run_num:2d}: score={solution.total_score:6.1f}, time={elapsed:6.2f}s")
    
    ga_total_time = time.time() - ga_start_time
    ga_scores = [r["score"] for r in ga_runs]
    ga_stats = {
        "best": max(ga_scores),
        "worst": min(ga_scores),
        "average": sum(ga_scores) / len(ga_scores),
        "median": sorted(ga_scores)[len(ga_scores)//2],
        "std_dev": (sum((s - sum(ga_scores)/len(ga_scores))**2 for s in ga_scores) / len(ga_scores))**0.5,
        "total_time": ga_total_time,
        "avg_time": ga_total_time / num_runs,
    }
    
    results["ga"] = {
        "runs": ga_runs,
        "statistics": ga_stats,
    }
    
    print(f"\n[GA] Statistics:")
    print(f"  Best:    {ga_stats['best']:.1f}")
    print(f"  Worst:   {ga_stats['worst']:.1f}")
    print(f"  Average: {ga_stats['average']:.1f}")
    print(f"  Std Dev: {ga_stats['std_dev']:.2f}")
    print(f"  Total time: {ga_total_time:.1f}s")
    
    # ========== RUN LOCAL SEARCH ==========
    print(f"\n[LS] Running Local Search ({num_runs} executions)...")
    ls_runs = []
    ls_start_time = time.time()
    
    for run_num in range(1, num_runs + 1):
        run_start = time.time()
        scheduler = LocalSearchScheduler(
            instance_data=instance_data,
            verbose=False,
            params={
                "MAX_ITERATIONS": 500,
                "RESTART_ITERATIONS": 50,
                "MAX_NO_IMPROVE": 100,
                "TIME_LIMIT": ls_time_limit,
                "RANDOM_RESTARTS": 3,
            },
        )
        solution = scheduler.generate_solution(start_time=run_start)
        elapsed = time.time() - run_start
        
        # Find corresponding GA score for comparison
        ga_score_for_run = ga_runs[run_num - 1]["score"]
        ls_score = solution.total_score
        improvement = ls_score - ga_score_for_run
        improvement_pct = (improvement / ga_score_for_run * 100) if ga_score_for_run > 0 else 0
        
        ls_runs.append({
            "run": run_num,
            "ga_score": ga_score_for_run,
            "ls_score": ls_score,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "time_seconds": elapsed,
            "programs_scheduled": len(solution.scheduled_programs),
        })
        print(f"  Run {run_num:2d}: GA={ga_score_for_run:6.1f} -> LS={ls_score:6.1f} "
              f"({improvement:+6.1f} / {improvement_pct:+6.2f}%), time={elapsed:6.2f}s")
    
    ls_total_time = time.time() - ls_start_time
    ls_scores = [r["ls_score"] for r in ls_runs]
    improvements = [r["improvement"] for r in ls_runs]
    improvements_pct = [r["improvement_pct"] for r in ls_runs]
    
    ls_stats = {
        "best": max(ls_scores),
        "worst": min(ls_scores),
        "average": sum(ls_scores) / len(ls_scores),
        "median": sorted(ls_scores)[len(ls_scores)//2],
        "std_dev": (sum((s - sum(ls_scores)/len(ls_scores))**2 for s in ls_scores) / len(ls_scores))**0.5,
        "total_time": ls_total_time,
        "avg_time": ls_total_time / num_runs,
        "improvement_avg": sum(improvements) / len(improvements),
        "improvement_avg_pct": sum(improvements_pct) / len(improvements_pct),
        "improvement_best": max(improvements),
        "improvement_worst": min(improvements),
        "improved_runs": sum(1 for imp in improvements if imp > 0),
    }
    
    results["ls"] = {
        "runs": ls_runs,
        "statistics": ls_stats,
    }
    
    print(f"\n[LS] Statistics:")
    print(f"  Best:    {ls_stats['best']:.1f}")
    print(f"  Worst:   {ls_stats['worst']:.1f}")
    print(f"  Average: {ls_stats['average']:.1f}")
    print(f"  Std Dev: {ls_stats['std_dev']:.2f}")
    print(f"  Total time: {ls_total_time:.1f}s")
    print(f"\n[LS] Improvements over GA:")
    print(f"  Average improvement: +{ls_stats['improvement_avg']:.1f} ({ls_stats['improvement_avg_pct']:+.2f}%)")
    print(f"  Best improvement:    +{ls_stats['improvement_best']:.1f}")
    print(f"  Worst improvement:   {ls_stats['improvement_worst']:+.1f}")
    print(f"  Improved runs:       {ls_stats['improved_runs']}/{num_runs}")
    
    # ========== RUN HYBRID GA+LS ==========
    print(f"\n[HYBRID] Running GA + Local Search ({num_runs} executions)...")
    hybrid_runs = []
    hybrid_start_time = time.time()
    
    half_time = max(10, ga_time_limit // 2)
    
    for run_num in range(1, num_runs + 1):
        run_start = time.time()
        
        # Run GA
        ga_start = time.time()
        ga_scheduler = GeneticScheduler(
            instance_data=instance_data,
            verbose=False,
            params={
                "POP_SIZE": 15,
                "GENERATIONS": 25,
                "CROSSOVER_RATE": 0.80,
                "MUTATION_RATE": 0.20,
                "TOURNAMENT_SIZE": 2,
                "ELITISM": 1,
                "TIME_LIMIT": half_time,
            },
        )
        ga_solution = ga_scheduler.generate_solution(start_time=ga_start)
        ga_time_used = time.time() - ga_start
        ga_score = ga_solution.total_score
        
        # Run LS on GA result
        ls_start = time.time()
        ls_scheduler = LocalSearchScheduler(
            instance_data=instance_data,
            verbose=False,
            params={
                "MAX_ITERATIONS": 500,
                "RESTART_ITERATIONS": 3,
                "MAX_NO_IMPROVE": 100,
                "TIME_LIMIT": half_time,
                "RANDOM_RESTARTS": 1,
            },
            initial_solution=ga_solution,
        )
        ls_solution = ls_scheduler.generate_solution(start_time=ls_start)
        ls_time_used = time.time() - ls_start
        final_score = ls_solution.total_score
        
        elapsed = time.time() - run_start
        
        # Calculate improvements
        ga_vs_hybrid = final_score - ga_score
        ga_vs_hybrid_pct = (ga_vs_hybrid / ga_score * 100) if ga_score > 0 else 0
        
        hybrid_runs.append({
            "run": run_num,
            "ga_score": ga_score,
            "hybrid_score": final_score,
            "improvement": ga_vs_hybrid,
            "improvement_pct": ga_vs_hybrid_pct,
            "ga_time": ga_time_used,
            "ls_time": ls_time_used,
            "total_time": elapsed,
            "programs_scheduled": len(ls_solution.scheduled_programs),
        })
        print(f"  Run {run_num:2d}: GA={ga_score:6.1f} -> Hybrid={final_score:6.1f} "
              f"({ga_vs_hybrid:+6.1f} / {ga_vs_hybrid_pct:+6.2f}%), "
              f"GA={ga_time_used:5.2f}s + LS={ls_time_used:5.2f}s")
    
    hybrid_total_time = time.time() - hybrid_start_time
    hybrid_scores = [r["hybrid_score"] for r in hybrid_runs]
    hybrid_improvements = [r["improvement"] for r in hybrid_runs]
    hybrid_improvements_pct = [r["improvement_pct"] for r in hybrid_runs]
    
    hybrid_stats = {
        "best": max(hybrid_scores),
        "worst": min(hybrid_scores),
        "average": sum(hybrid_scores) / len(hybrid_scores),
        "median": sorted(hybrid_scores)[len(hybrid_scores)//2],
        "std_dev": (sum((s - sum(hybrid_scores)/len(hybrid_scores))**2 for s in hybrid_scores) / len(hybrid_scores))**0.5,
        "total_time": hybrid_total_time,
        "avg_time": hybrid_total_time / num_runs,
        "improvement_avg": sum(hybrid_improvements) / len(hybrid_improvements),
        "improvement_avg_pct": sum(hybrid_improvements_pct) / len(hybrid_improvements_pct),
        "improvement_best": max(hybrid_improvements),
        "improvement_worst": min(hybrid_improvements),
        "improved_runs": sum(1 for imp in hybrid_improvements if imp > 0),
    }
    
    results["hybrid"] = {
        "runs": hybrid_runs,
        "statistics": hybrid_stats,
    }
    
    print(f"\n[HYBRID] Statistics:")
    print(f"  Best:    {hybrid_stats['best']:.1f}")
    print(f"  Worst:   {hybrid_stats['worst']:.1f}")
    print(f"  Average: {hybrid_stats['average']:.1f}")
    print(f"  Std Dev: {hybrid_stats['std_dev']:.2f}")
    print(f"  Total time: {hybrid_total_time:.1f}s")
    print(f"\n[HYBRID] Improvements over GA:")
    print(f"  Average improvement: +{hybrid_stats['improvement_avg']:.1f} ({hybrid_stats['improvement_avg_pct']:+.2f}%)")
    print(f"  Best improvement:    +{hybrid_stats['improvement_best']:.1f}")
    print(f"  Worst improvement:   {hybrid_stats['improvement_worst']:+.1f}")
    print(f"  Improved runs:       {hybrid_stats['improved_runs']}/{num_runs}")
    
    # ========== COMPARISON SUMMARY ==========
    print(f"\n{'='*100}")
    print("SUMMARY FOR THIS INSTANCE")
    print(f"{'='*100}")
    print(f"{'Algorithm':<15} {'Best':<10} {'Average':<10} {'Worst':<10} {'Std Dev':<10} {'Time':<10}")
    print(f"{'-'*100}")
    print(f"{'GA':<15} {ga_stats['best']:<10.1f} {ga_stats['average']:<10.1f} {ga_stats['worst']:<10.1f} {ga_stats['std_dev']:<10.2f} {ga_stats['total_time']:<10.1f}s")
    print(f"{'LS':<15} {ls_stats['best']:<10.1f} {ls_stats['average']:<10.1f} {ls_stats['worst']:<10.1f} {ls_stats['std_dev']:<10.2f} {ls_stats['total_time']:<10.1f}s")
    print(f"{'Hybrid':<15} {hybrid_stats['best']:<10.1f} {hybrid_stats['average']:<10.1f} {hybrid_stats['worst']:<10.1f} {hybrid_stats['std_dev']:<10.2f} {hybrid_stats['total_time']:<10.1f}s")
    print(f"{'-'*100}")
    print(f"{'LS vs GA':<15} {ls_stats['best']-ga_stats['best']:+10.1f} {ls_stats['average']-ga_stats['average']:+10.1f} {ls_stats['worst']-ga_stats['worst']:+10.1f}")
    print(f"{'  Improvement %':<15} {(ls_stats['best']-ga_stats['best'])/ga_stats['best']*100:+10.1f}% {(ls_stats['average']-ga_stats['average'])/ga_stats['average']*100:+10.1f}%")
    print(f"{'-'*100}")
    print(f"{'Hybrid vs GA':<15} {hybrid_stats['best']-ga_stats['best']:+10.1f} {hybrid_stats['average']-ga_stats['average']:+10.1f} {hybrid_stats['worst']-ga_stats['worst']:+10.1f}")
    print(f"{'  Improvement %':<15} {(hybrid_stats['best']-ga_stats['best'])/ga_stats['best']*100:+10.1f}% {(hybrid_stats['average']-ga_stats['average'])/ga_stats['average']*100:+10.1f}%")
    
    results["comparison_summary"] = {
        "ls_vs_ga": {
            "best_improvement": ls_stats['best'] - ga_stats['best'],
            "avg_improvement": ls_stats['average'] - ga_stats['average'],
            "worst_improvement": ls_stats['worst'] - ga_stats['worst'],
            "avg_improvement_pct": (ls_stats['average'] - ga_stats['average']) / ga_stats['average'] * 100,
        },
        "hybrid_vs_ga": {
            "best_improvement": hybrid_stats['best'] - ga_stats['best'],
            "avg_improvement": hybrid_stats['average'] - ga_stats['average'],
            "worst_improvement": hybrid_stats['worst'] - ga_stats['worst'],
            "avg_improvement_pct": (hybrid_stats['average'] - ga_stats['average']) / ga_stats['average'] * 100,
        },
    }
    
    return results


def run_all_instances(instances_dir: str, output_dir: str, num_runs: int = 10,
                      ga_time_limit: int = 60, ls_time_limit: int = 60,
                      verbose: bool = False):
    """Run analysis for all instances."""
    instances_path = Path(instances_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    instance_files = sorted(instances_path.glob("*.json"))
    
    if not instance_files:
        print(f"No instance files found in {instances_dir}")
        return
    
    print(f"\n{'='*100}")
    print("PER-INSTANCE OPTIMIZATION ANALYSIS")
    print(f"{'='*100}")
    print(f"Instances: {len(instance_files)}")
    print(f"Runs per instance: {num_runs}")
    print(f"GA time limit: {ga_time_limit}s")
    print(f"LS time limit: {ls_time_limit}s")
    
    all_results = []
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "num_instances": len(instance_files),
        "num_runs_per_instance": num_runs,
        "ga_time_limit": ga_time_limit,
        "ls_time_limit": ls_time_limit,
        "instances": [],
    }
    
    for i, instance_file in enumerate(instance_files, 1):
        instance_name = instance_file.stem
        
        try:
            result = run_instance_analysis(
                instance_name,
                str(instance_file),
                num_runs=num_runs,
                ga_time_limit=ga_time_limit,
                ls_time_limit=ls_time_limit,
                verbose=verbose,
            )
            
            if result:
                all_results.append(result)
                
                # Add to summary
                overall_summary["instances"].append({
                    "instance": instance_name,
                    "ga_avg": result["ga"]["statistics"]["average"],
                    "ls_avg": result["ls"]["statistics"]["average"],
                    "hybrid_avg": result["hybrid"]["statistics"]["average"],
                    "ls_improvement_avg": result["ls"]["statistics"]["improvement_avg"],
                    "ls_improvement_avg_pct": result["ls"]["statistics"]["improvement_avg_pct"],
                    "hybrid_improvement_avg": result["hybrid"]["statistics"]["improvement_avg"],
                    "hybrid_improvement_avg_pct": result["hybrid"]["statistics"]["improvement_avg_pct"],
                })
        
        except Exception as e:
            print(f"ERROR processing {instance_name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Save all results to JSON
    results_json_file = output_path / f"per_instance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_file, "w") as f:
        json.dump({
            "summary": overall_summary,
            "detailed": all_results,
        }, f, indent=2)
    print(f"\n\nDetailed results saved to: {results_json_file}")
    
    # Generate overall report
    report_file = output_path / f"per_instance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report = generate_overall_report(overall_summary, all_results)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Summary report saved to: {report_file}")
    
    print("\n" + report)


def generate_overall_report(summary: Dict, results: list) -> str:
    """Generate overall report for all instances."""
    report = []
    report.append("=" * 120)
    report.append("OVERALL ANALYSIS REPORT - PER-INSTANCE RESULTS")
    report.append("=" * 120)
    report.append(f"Date: {summary['timestamp']}")
    report.append(f"Total instances: {summary['num_instances']}")
    report.append(f"Runs per instance: {summary['num_runs_per_instance']}")
    report.append("")
    
    report.append("-" * 120)
    report.append("INSTANCE-BY-INSTANCE COMPARISON")
    report.append("-" * 120)
    report.append(f"{'Instance':<30} {'GA Avg':<12} {'LS Avg':<12} {'Hybrid Avg':<12} {'LS Imp%':<12} {'Hyb Imp%':<12}")
    report.append("-" * 120)
    
    total_ga = 0
    total_ls = 0
    total_hybrid = 0
    total_ls_imp_pct = 0
    total_hybrid_imp_pct = 0
    
    for inst in summary["instances"]:
        total_ga += inst["ga_avg"]
        total_ls += inst["ls_avg"]
        total_hybrid += inst["hybrid_avg"]
        total_ls_imp_pct += inst["ls_improvement_avg_pct"]
        total_hybrid_imp_pct += inst["hybrid_improvement_avg_pct"]
        
        report.append(f"{inst['instance']:<30} {inst['ga_avg']:<12.1f} {inst['ls_avg']:<12.1f} "
                     f"{inst['hybrid_avg']:<12.1f} {inst['ls_improvement_avg_pct']:<12.2f}% "
                     f"{inst['hybrid_improvement_avg_pct']:<12.2f}%")
    
    n = len(summary["instances"])
    
    # Handle case where no instances were successfully processed
    if n == 0:
        report.append("\nWARNING: No instances were successfully processed!")
        return "\n".join(report)
    
    avg_ga = total_ga / n
    avg_ls = total_ls / n
    avg_hybrid = total_hybrid / n
    avg_ls_imp_pct = total_ls_imp_pct / n
    avg_hybrid_imp_pct = total_hybrid_imp_pct / n
    
    report.append("-" * 120)
    report.append(f"{'AVERAGE':<30} {avg_ga:<12.1f} {avg_ls:<12.1f} {avg_hybrid:<12.1f} "
                 f"{avg_ls_imp_pct:<12.2f}% {avg_hybrid_imp_pct:<12.2f}%")
    report.append("=" * 120)
    
    # Summary statistics
    report.append("\nSUMMARY STATISTICS")
    report.append("-" * 120)
    report.append(f"LS (Local Search) vs GA (Genetic Algorithm):")
    report.append(f"  - Average improvement: +{(avg_ls - avg_ga):.1f} ({(avg_ls - avg_ga)/avg_ga*100:+.2f}%)")
    report.append(f"  - Positive improvements on {sum(1 for inst in summary['instances'] if inst['ls_improvement_avg_pct'] > 0)}/{n} instances")
    report.append("")
    report.append(f"Hybrid (GA+LS) vs GA (Genetic Algorithm):")
    report.append(f"  - Average improvement: +{(avg_hybrid - avg_ga):.1f} ({(avg_hybrid - avg_ga)/avg_ga*100:+.2f}%)")
    report.append(f"  - Positive improvements on {sum(1 for inst in summary['instances'] if inst['hybrid_improvement_avg_pct'] > 0)}/{n} instances")
    report.append("")
    
    # Best and worst performers
    best_ls_instance = max(summary["instances"], key=lambda x: x["ls_improvement_avg_pct"])
    worst_ls_instance = min(summary["instances"], key=lambda x: x["ls_improvement_avg_pct"])
    best_hybrid_instance = max(summary["instances"], key=lambda x: x["hybrid_improvement_avg_pct"])
    worst_hybrid_instance = min(summary["instances"], key=lambda x: x["hybrid_improvement_avg_pct"])
    
    report.append("BEST AND WORST PERFORMERS")
    report.append("-" * 120)
    report.append(f"LS best improvement: {best_ls_instance['instance']} ({best_ls_instance['ls_improvement_avg_pct']:+.2f}%)")
    report.append(f"LS worst improvement: {worst_ls_instance['instance']} ({worst_ls_instance['ls_improvement_avg_pct']:+.2f}%)")
    report.append(f"Hybrid best improvement: {best_hybrid_instance['instance']} ({best_hybrid_instance['hybrid_improvement_avg_pct']:+.2f}%)")
    report.append(f"Hybrid worst improvement: {worst_hybrid_instance['instance']} ({worst_hybrid_instance['hybrid_improvement_avg_pct']:+.2f}%)")
    
    report.append("\n" + "=" * 120)
    
    return "\n".join(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run per-instance optimization analysis"
    )
    parser.add_argument("--instances", default="data/input",
                        help="Directory with instance files")
    parser.add_argument("--output", default="results/per_instance",
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per instance")
    parser.add_argument("--ga-time", type=int, default=60,
                        help="Time limit for GA in seconds")
    parser.add_argument("--ls-time", type=int, default=60,
                        help="Time limit for LS in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    run_all_instances(
        instances_dir=args.instances,
        output_dir=args.output,
        num_runs=args.runs,
        ga_time_limit=args.ga_time,
        ls_time_limit=args.ls_time,
        verbose=args.verbose,
    )
