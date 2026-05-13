import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_ga_results(results_dir: str) -> Dict:
    """Load all GA result files."""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.startswith("ga_results_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    # Extract instance name from filename
                    # Format: ga_results_{instance}_{timestamp}.json
                    instance_name = filename.replace("ga_results_", "").rsplit("_", 2)[0]
                    
                    if instance_name not in results:
                        results[instance_name] = []
                    results[instance_name].append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results


def analyze_parameter_performance(results: Dict) -> Dict:
    """
    Analyze which parameters performed best across instances.
    
    Returns:
        Dict with parameter analysis and recommendations
    """
    param_stats = defaultdict(lambda: {
        "scores": [],
        "instances": [],
        "avg_score": 0,
        "std_dev": 0,
        "count": 0,
    })
    
    instance_param_map = {}  # Map instance -> list of (params, best_score)
    
    for instance_name, runs_list in results.items():
        instance_param_map[instance_name] = []
        
        for result_data in runs_list:
            params = result_data.get("parameters", {})
            stats = result_data.get("statistics", {})
            
            if not params or not stats:
                continue
            
            best_score = stats.get("best", 0)
            average_score = stats.get("average", 0)
            param_key = _params_to_key(params)
            
            param_stats[param_key]["scores"].append(best_score)
            param_stats[param_key]["instances"].append(instance_name)
            param_stats[param_key]["count"] += 1
            instance_param_map[instance_name].append((param_key, best_score))
    
    # Calculate statistics for each parameter set
    analysis = {}
    for param_key, stats_dict in param_stats.items():
        scores = stats_dict["scores"]
        if scores:
            avg = sum(scores) / len(scores)
            variance = sum((x - avg) ** 2 for x in scores) / len(scores)
            std_dev = variance ** 0.5
            
            analysis[param_key] = {
                "parameters": _key_to_params(param_key),
                "average_best_score": avg,
                "std_dev": std_dev,
                "num_instances": len(set(stats_dict["instances"])),
                "num_runs": stats_dict["count"],
                "best_individual_score": max(scores),
                "worst_individual_score": min(scores),
            }
    
    return analysis


def get_optimal_parameters(analysis: Dict, top_n: int = 5) -> List[Dict]:
    """
    Get the top N best parameter combinations ranked by average score.
    """
    ranked = sorted(
        analysis.items(),
        key=lambda x: x[1]["average_best_score"],
        reverse=True,
    )
    
    top_params = []
    for i, (key, stats) in enumerate(ranked[:top_n]):
        top_params.append({
            "rank": i + 1,
            "parameters": stats["parameters"],
            "average_score": stats["average_best_score"],
            "std_dev": stats["std_dev"],
            "num_instances": stats["num_instances"],
            "num_runs": stats["num_runs"],
            "score_range": f"{stats['worst_individual_score']:.1f}-{stats['best_individual_score']:.1f}",
        })
    
    return top_params


def analyze_parameter_sensitivity(analysis: Dict) -> Dict:
    """
    Analyze which individual parameters have the most impact.
    """
    sensitivity = {
        "population_size": defaultdict(list),
        "generations": defaultdict(list),
        "crossover_rate": defaultdict(list),
        "mutation_rate": defaultdict(list),
        "tournament_size": defaultdict(list),
        "elitism": defaultdict(list),
    }
    
    for param_key, stats in analysis.items():
        params = stats["parameters"]
        score = stats["average_best_score"]
        
        sensitivity["population_size"][params.get("POP_SIZE")].append(score)
        sensitivity["generations"][params.get("GENERATIONS")].append(score)
        sensitivity["crossover_rate"][params.get("CROSSOVER_RATE")].append(score)
        sensitivity["mutation_rate"][params.get("MUTATION_RATE")].append(score)
        sensitivity["tournament_size"][params.get("TOURNAMENT_SIZE")].append(score)
        sensitivity["elitism"][params.get("ELITISM")].append(score)
    
    sensitivity_summary = {}
    for param_name, param_values in sensitivity.items():
        param_summary = {}
        for value, scores in param_values.items():
            if scores:
                avg = sum(scores) / len(scores)
                param_summary[value] = {
                    "avg_score": avg,
                    "num_samples": len(scores),
                }
        
        if param_summary:
            best_value = max(param_summary.items(), key=lambda x: x[1]["avg_score"])
            sensitivity_summary[param_name] = {
                "best_value": best_value[0],
                "best_avg_score": best_value[1]["avg_score"],
                "all_values": param_summary,
            }
    
    return sensitivity_summary


def _params_to_key(params: Dict) -> str:
    """Convert parameters dict to a hashable key."""
    return f"POP{params.get('POP_SIZE')}_GEN{params.get('GENERATIONS')}_" \
           f"CR{params.get('CROSSOVER_RATE')}_MR{params.get('MUTATION_RATE')}_" \
           f"TS{params.get('TOURNAMENT_SIZE')}_EL{params.get('ELITISM')}"


def _key_to_params(key: str) -> Dict:
    """Convert key back to parameters dict."""
    parts = key.split("_")
    params = {}
    for part in parts:
        if part.startswith("POP"):
            params["POP_SIZE"] = int(part[3:])
        elif part.startswith("GEN"):
            params["GENERATIONS"] = int(part[3:])
        elif part.startswith("CR"):
            params["CROSSOVER_RATE"] = float(part[2:])
        elif part.startswith("MR"):
            params["MUTATION_RATE"] = float(part[2:])
        elif part.startswith("TS"):
            params["TOURNAMENT_SIZE"] = int(part[2:])
        elif part.startswith("EL"):
            params["ELITISM"] = int(part[2:])
    
    return params


def generate_analysis_report(results_dir: str) -> str:
    """Generate a comprehensive analysis report."""
    print("Loading GA results...")
    results = load_ga_results(results_dir)
    
    if not results:
        return "No GA results found in the specified directory."
    
    print(f"Loaded results for {len(results)} instances")
    
    print("\nAnalyzing parameter performance...")
    analysis = analyze_parameter_performance(results)
    
    print("Getting optimal parameters...")
    top_params = get_optimal_parameters(analysis, top_n=10)
    
    print("Analyzing parameter sensitivity...")
    sensitivity = analyze_parameter_sensitivity(analysis)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("GENETIC ALGORITHM PARAMETER ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Instances analyzed: {len(results)}")
    report.append(f"Parameter combinations tested: {len(analysis)}")
    report.append("")
    
    report.append("-" * 80)
    report.append("TOP 10 PARAMETER COMBINATIONS")
    report.append("-" * 80)
    for item in top_params:
        report.append(f"\nRank #{item['rank']}")
        report.append(f"  Average Score: {item['average_score']:.1f} (Â±{item['std_dev']:.1f})")
        report.append(f"  Score Range: {item['score_range']}")
        report.append(f"  Instances: {item['num_instances']}, Runs: {item['num_runs']}")
        params = item["parameters"]
        report.append(f"  Parameters:")
        report.append(f"    - Population Size: {params.get('POP_SIZE')}")
        report.append(f"    - Generations: {params.get('GENERATIONS')}")
        report.append(f"    - Crossover Rate: {params.get('CROSSOVER_RATE')}")
        report.append(f"    - Mutation Rate: {params.get('MUTATION_RATE')}")
        report.append(f"    - Tournament Size: {params.get('TOURNAMENT_SIZE')}")
        report.append(f"    - Elitism: {params.get('ELITISM')}")
    
    report.append("")
    report.append("-" * 80)
    report.append("PARAMETER SENSITIVITY ANALYSIS")
    report.append("-" * 80)
    for param_name, summary in sensitivity.items():
        report.append(f"\n{param_name.upper().replace('_', ' ')}")
        report.append(f"  Optimal value: {summary['best_value']} (avg score: {summary['best_avg_score']:.1f})")
        report.append(f"  All tested values:")
        for value, stats in sorted(summary["all_values"].items()):
            report.append(f"    {value}: {stats['avg_score']:.1f} (n={stats['num_samples']})")
    
    report.append("")
    report.append("-" * 80)
    report.append("RECOMMENDATIONS FOR OPTIMAL CONFIGURATION")
    report.append("-" * 80)
    
    best_config = top_params[0]
    report.append(f"\nBased on analysis, the recommended configuration is:")
    report.append(f"  POP_SIZE: {best_config['parameters'].get('POP_SIZE')}")
    report.append(f"  GENERATIONS: {best_config['parameters'].get('GENERATIONS')}")
    report.append(f"  CROSSOVER_RATE: {best_config['parameters'].get('CROSSOVER_RATE')}")
    report.append(f"  MUTATION_RATE: {best_config['parameters'].get('MUTATION_RATE')}")
    report.append(f"  TOURNAMENT_SIZE: {best_config['parameters'].get('TOURNAMENT_SIZE')}")
    report.append(f"  ELITISM: {best_config['parameters'].get('ELITISM')}")
    report.append(f"\nExpected average score: {best_config['average_score']:.1f}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    report = generate_analysis_report(str(results_dir))
    print(report)
    
    # Save report
    output_file = results_dir / "parameter_analysis_report.txt"
    with open(output_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_file}")
