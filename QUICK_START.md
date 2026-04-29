# QUICK START GUIDE - RUN NOW

## ⚡ 5-Minute Quick Start

### Option 1: Test with One Instance (2-3 minutes)
```bash
python main_new.py --scheduler 3 --ga-config experiment_2_medium_pop --runs 10
```
Then select an instance from the menu (e.g., australia_iptv.json)

### Option 2: Run All Instances (1.5 hours)
```bash
python run_all_experiments.py
```
Runs all 17 instances through all 4 configurations

---

## 📖 Understanding the Output

When you run the script, you'll see:

```
==================================================================================
Running Genetic Algorithm (10 runs)
Configuration: experiment_2_medium_pop
Parameters: POP_SIZE=10, GENERATIONS=30, CROSSOVER=0.8, MUTATION=0.15
==================================================================================

Run 1/10... Score:  2150 | Time:   4.32s
Run 2/10... Score:  2145 | Time:   4.28s
...
Run 10/10... Score:  2147 | Time:   4.30s

================================================================================
Results Summary:
  Best:    2155
  Worst:   2145
  Average: 2150.0
  Total time: 43.1s
================================================================================
```

**What it means:**
- **Score**: The quality of the solution (higher is better)
- **Time**: How long that run took
- **Best**: Best score from all 10 runs
- **Average**: Mean score (expected performance)
- **Total time**: Should be < 5 minutes per instance

---

## 🔍 What's Happening Inside

### The Algorithm Flow

```
GENERATION 0 (Init):
  Create 10 individuals
    ↓
  Best: 2093 (from Greedy heuristic)
  
GENERATION 1:
  Keep 1 best (2093)
  Create 9 new ones through crossover+mutation
    ↓
  Best: 2098 (improved!)
  
GENERATION 2:
  Keep 1 best (2098)
  Create 9 new ones
    ↓
  Best: 2101 (improved again!)
  
... repeat 30 times total ...

FINAL:
  Best found: 2120 (+27 improvement from start)
```

---

## ⚙️ The 4 Configurations

### Config 1: Small Population (Fast)
- **When**: Quick testing
- **Time**: 2-3 min per instance
- **Quality**: Good
- **Run**: `--ga-config experiment_1_small_pop`

### Config 2: Medium Population (RECOMMENDED) ⭐
- **When**: Regular use
- **Time**: 4-5 min per instance
- **Quality**: Excellent
- **Run**: `--ga-config experiment_2_medium_pop`

### Config 3: Large Population (Thorough)
- **When**: Need best quality
- **Time**: 5-6 min per instance
- **Quality**: Best
- **Run**: `--ga-config experiment_3_large_pop`

### Config 4: High Mutation (Exploration)
- **When**: Exploring solution space
- **Time**: 4-5 min per instance
- **Quality**: Good
- **Run**: `--ga-config experiment_4_high_mutation`

---

## 📊 Example Results (What You'll Get)

### After Running All 17 Instances with Config 2:

| Instance | Best | Average | Std Dev | Time |
|----------|------|---------|---------|------|
| australia_iptv | 2155 | 2150.0 | ±3.2 | 4.3s |
| canada_pw | 3210 | 3205.5 | ±2.8 | 4.5s |
| france_iptv | 2890 | 2885.0 | ±3.5 | 4.4s |
| ... | ... | ... | ... | ... |

**Interpretation:**
- **Best**: Best score in 10 runs
- **Average**: Expected score
- **Std Dev**: How consistent (±3 = very consistent)
- **Time**: How long each run took

---

## 🎯 Quick Comparison

### What to Compare Between Configurations:

1. **Quality** (average score): Higher = better
2. **Consistency** (std dev): Lower = more stable
3. **Time** (time per run): Should all fit in 5 minutes
4. **Overall**: Best config = highest avg + lowest std dev + fast enough

### Example:
```
Config 1: Average 2148.5 (±4.2) - 2.5 min/instance
Config 2: Average 2150.0 (±3.2) - 4.3 min/instance ✓ BEST
Config 3: Average 2150.5 (±3.5) - 5.5 min/instance (too slow)
Config 4: Average 2149.0 (±5.1) - 4.2 min/instance
```
→ Choose Config 2

---

## 📝 Documenting Your Results

### Step 1: Run One Configuration
```bash
python main_new.py --scheduler 3 --ga-config experiment_2_medium_pop --runs 10
```

### Step 2: Note the Output
Copy the results (best, average, total time)

### Step 3: Fill in Template
Edit `RESULTS_README_TEMPLATE.md` with your numbers

### Step 4: Repeat for Other Configs
Run configs 1, 3, and 4 and collect results

### Step 5: Compare
Which config gives best average score + stable results?

---

## ⏱️ Time Management

### Expected Times

For **single instance**:
- Config 1: 2-3 minutes
- Config 2: 4-5 minutes (recommended)
- Config 3: 5-6 minutes
- Config 4: 4-5 minutes

For **all 17 instances with Config 2**:
- 17 × 4.3 min ≈ 73 minutes (1.2 hours)

For **all 4 configs on all instances**:
- 17 × 4 configs × 4 min ≈ 272 minutes (4.5 hours)

---

## ❓ Troubleshooting

### If Script Crashes
```bash
# Check syntax
python -m py_compile scheduler/genetic_algorithm.py

# Run with verbose
python main_new.py --scheduler 3 --verbose
```

### If Scores Are Not Improving
- Try Config 3 (larger population)
- Or Config 4 (higher mutation)

### If Results Are Too Slow
- Use Config 1 for quick tests
- Or reduce `--runs` parameter

### If Results Show Same Score Every Time
- Algorithm may have converged
- Try higher mutation (Config 4)
- Or increase generations

---

## ✅ Checklist Before Submission

- [ ] Run all 4 configurations on representative instances
- [ ] Compare results (which config is best?)
- [ ] Fill in `RESULTS_README_TEMPLATE.md`
- [ ] Verify times are < 5 min per instance
- [ ] Save output files (data/output/)
- [ ] Commit files to repo:
  - `scheduler/genetic_algorithm.py` (improved)
  - `scheduler/GA_CONFIG.json` (configurations)
  - `GA_GUIDE.md` (full documentation)
  - `IMPLEMENTATION_SUMMARY.md` (this summary)
  - `RESULTS_README.md` (your results)
  - `main_new.py` (enhanced main)

---

## 🚀 Ready to Go!

### Quick Test (2 minutes)
```bash
python main_new.py --scheduler 3 --runs 5
```

### Full Experiment (5 hours)
```bash
python run_all_experiments.py
```

### Check Results
```bash
# Look at output files
ls data/output/
```

---

*Last Updated: 2026-04-27*  
*Status: ✅ Ready to Run*  
*Deadline: April 30*
