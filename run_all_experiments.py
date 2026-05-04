"""
Script to run all experiments and collect results
Run all instances through all GA configurations
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration to run
CONFIGS = [
    "experiment_1_small_pop",
    "experiment_2_medium_pop", 
    "experiment_3_large_pop",
    "experiment_4_high_mutation"
]

RUNS_PER_INSTANCE = 10

def get_instances():
    """Get all input instances"""
    input_dir = Path("data/input")
    return sorted([f.name for f in input_dir.glob("*.json")])


def run_instance(instance_name, config, runs):
    """Run single instance with config"""
    print(f"\n{'='*80}")
    print(f"Instance: {instance_name}")
    print(f"Config: {config}")
    print(f"Runs: {runs}")
    print(f"{'='*80}")
    
    cmd = [
        "python", "main_new.py",
        "--input", f"data/input/{instance_name}",
        "--scheduler", "3",
        "--ga-config", config,
        "--runs", str(runs)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            'status': 'success',
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'stdout': 'Process timed out after 5 minutes',
            'stderr': ''
        }
    except Exception as e:
        return {
            'status': 'error',
            'stdout': '',
            'stderr': str(e)
        }


def main():
    instances = get_instances()
    
    print(f"\n{'='*80}")
    print(f"GENETIC ALGORITHM BATCH RUNNER")
    print(f"{'='*80}")
    print(f"Instances found: {len(instances)}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Runs per instance: {RUNS_PER_INSTANCE}")
    print(f"Total runs: {len(instances) * len(CONFIGS) * RUNS_PER_INSTANCE}")
    print(f"Estimated time: ~{len(instances) * len(CONFIGS) * 5 / 60:.1f} hours")
    print(f"{'='*80}\n")
    
    input("Press Enter to start running all experiments...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'instances': instances,
        'configs': CONFIGS,
        'runs_per_instance': RUNS_PER_INSTANCE,
        'results': {}
    }
    
    total_start = time.time()
    
    for instance in instances:
        results['results'][instance] = {}
        
        for config in CONFIGS:
            print(f"\nProgress: {instance} with {config}...")
            
            start = time.time()
            result = run_instance(instance, config, RUNS_PER_INSTANCE)
            elapsed = time.time() - start
            
            results['results'][instance][config] = {
                'status': result['status'],
                'time': elapsed,
                'timestamp': datetime.now().isoformat()
            }
            
            if result['status'] == 'success':
                print(f"✓ Completed in {elapsed:.1f}s")
            else:
                print(f"✗ Failed: {result['status']}")
                print(result['stderr'][:200] if result['stderr'] else 'No error details')
    
    total_elapsed = time.time() - total_start
    
    # Save results
    results['total_time'] = total_elapsed
    results_file = Path("results") / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH RUN COMPLETE")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
