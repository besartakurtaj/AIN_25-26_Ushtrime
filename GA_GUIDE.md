# GENETIC ALGORITHM SCHEDULER - COMPLETE GUIDE

## Overview

This is a **Genetic Algorithm** implementation for TV schedule optimization. The algorithm evolves a population of solutions over multiple generations to find better scheduling solutions.

---

## 1. ALGORITHM EXPLANATION

### What Happens at Each Step

#### Step 1: INITIALIZATION
- **Input**: One instance (e.g., australia_iptv.json)
- **Process**:
  1. Create 1st individual: Run Greedy+Lookahead heuristic → Solution #1 (e.g., score 2093)
  2. Create individuals 2 to POP_SIZE: Mutate solution #1 → Creates different variations
- **Output**: Population of POP_SIZE individuals (e.g., 10 solutions with different scores)
- **Example** (POP_SIZE=10):
  - Individual 1: 2093 (from Greedy+Lookahead)
  - Individual 2: 2087 (mutated version)
  - Individual 3: 2091 (mutated version)
  - ... (7 more individuals)

#### Step 2: TOURNAMENT SELECTION
- **Purpose**: Pick parents for reproduction
- **Process**:
  1. Randomly pick TOURNAMENT_SIZE individuals (e.g., 2)
  2. Select the one with best fitness (highest score)
- **Result**: Biases selection toward better solutions while preserving diversity

#### Step 3: CROSSOVER (Genetic Recombination)
- **Purpose**: Combine two parents to create a new child
- **Probability**: CROSSOVER_RATE (default 0.8 = 80%)
- **Two methods**:

  **Method A: Time Segment Crossover**
  - Split at random time point
  - Child gets: early programs from Parent1 + late programs from Parent2
  
  **Method B: Channel-based Crossover**
  - Child gets: random channels from Parent1 + other channels from Parent2

- **Result**: NEW solution that inherits good genes from both parents

#### Step 4: MUTATION (Local Search/Exploration)
- **Purpose**: Create variation by small changes
- **Probability**: MUTATION_RATE (default 0.15 = 15%)
- **Two methods**:

  **Method A: Reverse Segment**
  - Reverse order of a random segment
  - Example: [A,B,C,D,E] → [A,D,C,B,E]
  
  **Method B: Displacement**
  - Move one program to random position
  - Example: [A,B,C,D] → [C,A,B,D]

- **Result**: MODIFIED version of child with small changes

#### Step 5: ELITISM & REPLACEMENT
- **Elitism**: Keep top ELITISM individuals (default 1)
  - These are the best solutions from previous generation
  - Guaranteed to be in next generation
  - Prevents loss of good solutions
- **Replacement**: 
  - Discard: All individuals except the ELITISM best ones
  - Keep: Top ELITISM individuals + new children from crossover/mutation
- **Population size**: Always stays at POP_SIZE

#### Step 6: REPEAT
- Go back to step 2
- Repeat for GENERATIONS times (default 30)
- Track best solution found so far

### Timeline Example (POP_SIZE=10, australia_iptv)

```
GENERATION 0 (Initialization):
  Population: [2093, 2087, 2091, 2090, 2088, 2089, 2085, 2092, 2086, 2088]
  Best so far: 2093

GENERATION 1:
  Elitism: Keep 1 best → [2093]
  Create 9 new children through crossover+mutation
  New population: [2093, 2094, 2095, 2091, 2090, 2089, 2092, 2091, 2093, 2090]
  Best so far: 2095 (improved!)

GENERATION 2:
  Elitism: Keep 1 best → [2095]
  Create 9 new children
  New population: [2095, 2096, 2094, 2093, 2095, 2097, 2092, 2095, 2091, 2094]
  Best so far: 2097 (improved!)
  
  ... (continues for 30 generations)

GENERATION 30 (Final):
  Best found: 2150
```

### What Happens to Individuals

| Type | Action | Generation |
|------|--------|-----------|
| Old individuals (not in top ELITISM) | **DISCARDED** after selection | Every generation |
| Individuals from crossover | **ENTER** population to compete | Every generation |
| Individuals from mutation | **ENTER** population to compete | Every generation |
| Best individuals (ELITISM) | **PRESERVED** to next generation | Every generation |
| **Population size** | **ALWAYS** POP_SIZE (constant) | Every generation |

---

## 2. PARAMETERS

All parameters are defined in `scheduler/genetic_algorithm.py`:

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **POP_SIZE** | 10 | 5-30 | Population size. Larger = more diversity, slower |
| **GENERATIONS** | 30 | 10-50 | How many generations to evolve. More = better but slower |
| **CROSSOVER_RATE** | 0.8 | 0.5-1.0 | Probability of crossover. Higher = more combination |
| **MUTATION_RATE** | 0.15 | 0.05-0.5 | Probability of mutation. Higher = more exploration |
| **TOURNAMENT_SIZE** | 2 | 2-5 | Tournament competitors. Larger = stronger selection |
| **ELITISM** | 1 | 1-3 | Best individuals to preserve. Higher = more stability |

### Parameter Effects

#### POP_SIZE
- **5**: Fast, may miss good solutions
- **10**: Balanced (recommended)
- **20**: Slow, more thorough exploration
- **30**: Very slow, diminishing returns

#### GENERATIONS
- **10**: Quick, may converge early
- **30**: Balanced (recommended)
- **50**: Very thorough but slow

#### CROSSOVER_RATE
- **0.5**: Low - mostly copy parents
- **0.8**: High - more mixing (recommended)
- **1.0**: Always crossover

#### MUTATION_RATE
- **0.05**: Very low - mostly stability
- **0.15**: Medium (recommended)
- **0.3**: High - more exploration
- **0.5**: Very high - chaotic

#### TOURNAMENT_SIZE
- **2**: Loose selection - high diversity
- **3**: Medium selection
- **5**: Strong selection - fast convergence but risks local optima

---

## 3. EXPERIMENTS & RESULTS

### Experiment Configurations (GA_CONFIG.json)

Four pre-defined configurations are provided:

#### Experiment 1: Small Population (5)
```json
{
  "POP_SIZE": 5,
  "GENERATIONS": 30,
  "CROSSOVER_RATE": 0.8,
  "MUTATION_RATE": 0.15,
  "TOURNAMENT_SIZE": 2,
  "ELITISM": 1
}
```
- **Best for**: Quick tests, small instances
- **Time**: ~2-3 minutes per instance
- **Trade-off**: Fast but may miss good solutions

#### Experiment 2: Medium Population (10) - RECOMMENDED
```json
{
  "POP_SIZE": 10,
  "GENERATIONS": 30,
  "CROSSOVER_RATE": 0.8,
  "MUTATION_RATE": 0.15,
  "TOURNAMENT_SIZE": 2,
  "ELITISM": 1
}
```
- **Best for**: Balanced performance
- **Time**: ~4-5 minutes per instance
- **Trade-off**: Good results without being too slow

#### Experiment 3: Large Population (20)
```json
{
  "POP_SIZE": 20,
  "GENERATIONS": 20,
  "CROSSOVER_RATE": 0.8,
  "MUTATION_RATE": 0.15,
  "TOURNAMENT_SIZE": 3,
  "ELITISM": 2
}
```
- **Best for**: Large instances, more thorough search
- **Time**: ~5+ minutes per instance
- **Trade-off**: Better quality, slower

#### Experiment 4: High Mutation Rate (30%)
```json
{
  "POP_SIZE": 10,
  "GENERATIONS": 30,
  "CROSSOVER_RATE": 0.7,
  "MUTATION_RATE": 0.3,
  "TOURNAMENT_SIZE": 2,
  "ELITISM": 1
}
```
- **Best for**: Exploring more solutions
- **Time**: ~4-5 minutes per instance
- **Trade-off**: More exploration, less exploitation

---

## 4. HOW TO RUN

### Running with Default Parameters (POP_SIZE=10)

```bash
python main.py
# Then select: 3 (Genetic Algorithm)
```

### Running with Specific Configuration

```bash
python main_new.py --scheduler 3 --ga-config experiment_2_medium_pop --runs 10
```

### Running with Custom Number of Runs

```bash
python main_new.py --scheduler 3 --runs 20
```

### Running All Instances with All Configurations

```bash
# Quick test with small population
for file in data/input/*.json; do
    python main.py --input "$file" --scheduler 3 --ga-config experiment_1_small_pop --runs 5
done

# Full test with recommended population
for file in data/input/*.json; do
    python main_new.py --input "$file" --scheduler 3 --ga-config experiment_2_medium_pop --runs 10
done
```

---

## 5. UNDERSTANDING RESULTS

### Output Example

```
==================================================================================
Running Genetic Algorithm (10 runs)
Configuration: experiment_2_medium_pop
Parameters: POP_SIZE=10, GENERATIONS=30, CROSSOVER=0.8, MUTATION=0.15
==================================================================================

Run 1/10... Score:  2150 | Time:   4.32s
Run 2/10... Score:  2145 | Time:   4.28s
Run 3/10... Score:  2152 | Time:   4.35s
Run 4/10... Score:  2148 | Time:   4.30s
Run 5/10... Score:  2155 | Time:   4.33s
Run 6/10... Score:  2150 | Time:   4.31s
Run 7/10... Score:  2149 | Time:   4.32s
Run 8/10... Score:  2153 | Time:   4.29s
Run 9/10... Score:  2151 | Time:   4.34s
Run 10/10... Score:  2147 | Time:   4.30s

================================================================================
Results Summary:
  Best:    2155
  Worst:   2145
  Average: 2150.0
  Total time: 43.1s
================================================================================
```

### Interpreting Statistics

- **Best**: Maximum score across all runs (best case)
- **Worst**: Minimum score across all runs (worst case)
- **Average**: Mean of all runs (expected performance)
- **Std Dev**: How consistent results are (low = stable)
  - Low std dev (±5) = consistent algorithm
  - High std dev (±20) = unstable results
- **Total Time**: Sum of all run times (must be ≤ 5 min per instance)

---

## 6. TIPS FOR PARAMETER TUNING

### If Algorithm is Too Slow
- ↓ POP_SIZE (reduce from 20 to 10)
- ↓ GENERATIONS (reduce from 50 to 30)
- ↑ MUTATION_RATE (explore faster)

### If Results Are Not Improving
- ↑ POP_SIZE (more individuals = more diversity)
- ↑ GENERATIONS (more time to evolve)
- ↓ MUTATION_RATE (focus on good solutions)
- ↑ ELITISM (preserve best solutions)

### If Results Are Too Random (high std dev)
- ↑ POP_SIZE (more stable population)
- ↓ MUTATION_RATE (less chaotic changes)
- ↑ TOURNAMENT_SIZE (stronger selection pressure)
- ↑ ELITISM (preserve consistency)

### If Results Are Converging Too Early
- ↑ MUTATION_RATE (more exploration)
- ↓ TOURNAMENT_SIZE (less selection pressure)
- ↑ POP_SIZE (more diversity)

---

## 7. FILE STRUCTURE

```
scheduler/
├── genetic_algorithm.py          # GA implementation (fully documented)
├── GA_CONFIG.json                # Parameter configurations for experiments
├── greedy_lookahead_scheduler.py # Base heuristic for initialization
└── ...

data/
├── input/                        # Input instances (17 files)
│   ├── australia_iptv.json
│   ├── canada_pw.json
│   └── ...
├── output/                       # Output solutions
│   └── ...
└── initial_solutions/            # Stored initial solutions (optional)
    ├── australia_iptv_initial.json
    └── ...

results/
├── experiment_1_results.json     # Results from experiment 1
├── experiment_2_results.json     # Results from experiment 2
├── experiment_3_results.json     # Results from experiment 3
├── experiment_4_results.json     # Results from experiment 4
└── summary_README.md             # Final summary of all experiments

main.py                           # Original main script
main_new.py                       # Enhanced main script with config support
```

---

## 8. REQUIREMENTS & TIME MANAGEMENT

### Time Budget
- **Per instance**: ~5 minutes maximum
- **All 17 instances**: ~85 minutes (1.5 hours)
- **Per run**: 30-60 seconds depending on instance size

### Execution Example (10 runs × 17 instances)
```
Experiment 2 (POP_SIZE=10):
- Per instance: 4-5 minutes (10 runs)
- All instances: 68-85 minutes
- Total: ~1.5 hours
```

### Tips to Fit in Time Budget
1. Use `POP_SIZE=10` with `GENERATIONS=30` (4-5 min per instance)
2. Parallel processing (if multi-core available)
3. Start with smaller population for initial tests
4. Track time to ensure ≤ 5 minutes per instance

---

## 9. QUESTIONS & ANSWERS

### Q: Why do different runs have different scores?
**A**: Because of random selection, crossover, and mutation. Each run evolves differently due to randomness.

### Q: What does "10 individuals from mutation" vs "from crossover" mean?
**A**: Each generation creates POP_SIZE new individuals:
- Some from crossover (combining two parents)
- Some from mutation (modifying existing solutions)
- Some preserved through elitism (best from last gen)

### Q: How many solutions should I generate?
**A**: Run each configuration 10 times per instance to get reliable statistics.

### Q: What if all runs get the same score?
**A**: Algorithm is converging. Either increase MUTATION_RATE or POP_SIZE.

### Q: How to choose the best configuration?
**A**: Compare average scores across experiments. Best = highest average with stable std dev.

---

## 10. NEXT STEPS

1. **Run Experiment 2** (recommended): `python main_new.py --scheduler 3 --ga-config experiment_2_medium_pop --runs 10`
2. **Collect results** from all 17 instances
3. **Run other experiments** to compare
4. **Analyze statistics**: Which configuration performs best?
5. **Document findings** in results_README.md
6. **Submit all files** with results and parameters

---

## 11. REFERENCES

This genetic algorithm uses:
- **Tournament Selection**: Unbiased, preserves diversity
- **Two Crossover Methods**: Time-based and Channel-based recombination
- **Two Mutation Methods**: Reversals and Displacements
- **Elitism**: Preserves best solutions across generations
- **Greedy Initialization**: Fast, high-quality starting solution

---

*Genetic Algorithm Scheduler v1.0*  
*Last Updated: 2026-04-27*  
*Implementation Complete with Full Documentation*
