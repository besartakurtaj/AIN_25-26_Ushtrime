"""
TV Schedule Optimizer - Main Script
Supports Beam Search, Greedy+Lookahead, and Genetic Algorithm.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.greedy_lookahead_scheduler import GreedyLookaheadScheduler
from scheduler.genetic_algorithm import GeneticScheduler
from utils.utils import Utils


DEFAULT_GA_CONFIGS = {
    "experiment_1_small_pop": {
        "POP_SIZE": 5,
        "GENERATIONS": 30,
        "CROSSOVER_RATE": 0.80,
        "MUTATION_RATE": 0.25,
        "TOURNAMENT_SIZE": 2,
        "ELITISM": 1,
        "TIME_LIMIT": 300,
    },
    "experiment_2_medium_pop": {
        "POP_SIZE": 10,
        "GENERATIONS": 30,
        "CROSSOVER_RATE": 0.80,
        "MUTATION_RATE": 0.25,
        "TOURNAMENT_SIZE": 3,
        "ELITISM": 1,
        "TIME_LIMIT": 300,
    },
    "experiment_3_large_pop": {
        "POP_SIZE": 20,
        "GENERATIONS": 20,
        "CROSSOVER_RATE": 0.85,
        "MUTATION_RATE": 0.20,
        "TOURNAMENT_SIZE": 3,
        "ELITISM": 2,
        "TIME_LIMIT": 300,
    },
    "experiment_4_high_mutation": {
        "POP_SIZE": 10,
        "GENERATIONS": 30,
        "CROSSOVER_RATE": 0.70,
        "MUTATION_RATE": 0.40,
        "TOURNAMENT_SIZE": 2,
        "ELITISM": 1,
        "TIME_LIMIT": 300,
    },
}


def load_ga_config(config_name: str | None) -> dict:
    config_path = Path("scheduler/GA_CONFIG.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if config_name and config_name in data:
            cfg = data[config_name]
            return cfg.get("params", cfg)

    if config_name and config_name in DEFAULT_GA_CONFIGS:
        return DEFAULT_GA_CONFIGS[config_name]
    return DEFAULT_GA_CONFIGS["experiment_2_medium_pop"]


def run_genetic_algorithm(instance, num_runs: int, config_name: str | None, verbose: bool):
    params = load_ga_config(config_name)
    results = []
    best_solution = None

    print("\n" + "=" * 80)
    print(f"Running Genetic Algorithm | runs={num_runs} | config={config_name or 'experiment_2_medium_pop'}")
    print(
        f"Parameters: POP_SIZE={params['POP_SIZE']}, GENERATIONS={params['GENERATIONS']}, "
        f"CROSSOVER_RATE={params['CROSSOVER_RATE']}, MUTATION_RATE={params['MUTATION_RATE']}, "
        f"TOURNAMENT_SIZE={params['TOURNAMENT_SIZE']}, ELITISM={params['ELITISM']}"
    )
    print("=" * 80)

    for run in range(1, num_runs + 1):
        run_params = params.copy()
        run_params["SEED"] = int(time.time_ns() % 2_147_483_647) + run

        start = time.time()
        scheduler = GeneticScheduler(instance_data=instance, verbose=verbose, params=run_params)
        solution = scheduler.generate_solution()
        elapsed = time.time() - start

        results.append({"run": run, "score": solution.total_score, "time_seconds": elapsed})
        if best_solution is None or solution.total_score > best_solution.total_score:
            best_solution = solution

        print(f"Run {run:2d}/{num_runs} | Score: {solution.total_score:5d} | Time: {elapsed:7.2f}s | Seed: {run_params['SEED']}")

    scores = [r["score"] for r in results]
    times = [r["time_seconds"] for r in results]
    avg = sum(scores) / len(scores)
    std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0

    print("\nResults Summary")
    print(f"Best: {max(scores)} | Worst: {min(scores)} | Average: {avg:.1f} | Std: {std:.1f}")
    print(f"Total time: {sum(times):.2f}s | Avg time/run: {sum(times)/len(times):.2f}s")

    return best_solution, {
        "parameters": params,
        "runs": results,
        "statistics": {
            "best": max(scores),
            "worst": min(scores),
            "average": avg,
            "std_dev": std,
            "total_time": sum(times),
        },
    }


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file", help="Path to input JSON")
    parser_arg.add_argument("--scheduler", "-s", choices=["1", "2", "3"], default=None,
                            help="1=Beam, 2=Greedy+Lookahead, 3=Genetic")
    parser_arg.add_argument("--runs", "-r", type=int, default=10, help="GA runs, default 10")
    parser_arg.add_argument("--ga-config", "-c", default="experiment_2_medium_pop")
    parser_arg.add_argument("--verbose", "-v", action="store_true")
    args = parser_arg.parse_args()

    file_path = args.input_file if args.input_file else select_file()
    parser = Parser(file_path)
    instance = parser.parse()
    Utils.set_current_instance(instance)

    print("\nInstance information:")
    print(f"  File: {file_path}")
    print(f"  Opening time: {instance.opening_time}")
    print(f"  Closing time: {instance.closing_time}")
    print(f"  Total channels: {len(instance.channels)}")

    choice = args.scheduler
    if not choice:
        print("\nChoose scheduler:")
        print("1: Beam Search")
        print("2: Greedy + Lookahead")
        print("3: Genetic Algorithm")
        choice = input("Select scheduler [1/2/3] (default 1): ").strip() or "1"

    if choice == "3":
        best_solution, experiment_results = run_genetic_algorithm(
            instance=instance,
            num_runs=args.runs,
            config_name=args.ga_config,
            verbose=args.verbose,
        )
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        result_file = results_dir / f"ga_results_{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2)
        print(f"Results JSON saved: {result_file}")
    elif choice == "2":
        scheduler = GreedyLookaheadScheduler(instance_data=instance, lookahead_limit=4, density_percentile=25, verbose=False)
        solution = scheduler.generate_random_solution()
    else:
        scheduler = BeamSearchScheduler(instance_data=instance, beam_width=100, lookahead_limit=4, density_percentile=25, verbose=False)
        solution = scheduler.generate_random_solution()

    print(f"\nGenerated solution with total score: {best_solution.total_score}")
    algorithm_name = "geneticscheduler" if choice == "3" else type(scheduler).__name__.lower()
    serializer = SolutionSerializer(input_file_path=file_path, algorithm_name=algorithm_name)
    serializer.serialize(best_solution)
    print("✓ Best solution saved to output file")


if __name__ == "__main__":
    main()
