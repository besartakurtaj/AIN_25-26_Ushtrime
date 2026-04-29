"""
GENETIC ALGORITHM SCHEDULER
============================

Real GA implementation for TV scheduling.

Key behavior:
- Initial population contains one Greedy+Lookahead solution and POP_SIZE-1 stochastic variants.
- Selection uses tournament selection.
- Crossover creates a new child from two parents using a random time split.
- Mutation changes a solution by cutting it at a random point and refilling the rest stochastically.
- Elitism keeps the best old individuals; the rest of the old population is replaced by children.
- Every run uses a different RNG seed unless a SEED parameter is provided.
"""

import bisect
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional

from models.solution import Solution
from models.schedule import Schedule
from scheduler.greedy_lookahead_scheduler import GreedyLookaheadScheduler


class GeneticScheduler:
    DEFAULT_PARAMS = {
        "POP_SIZE": 10,
        "GENERATIONS": 30,
        "CROSSOVER_RATE": 0.90,
        "MUTATION_RATE": 0.50,
        "TOURNAMENT_SIZE": 2,
        "ELITISM": 1,
        "TIME_LIMIT": 300,
        "LOOKAHEAD_LIMIT": 4,
        "DENSITY_PERCENTILE": 25,
        "RANDOM_TOP_K": 10,
        "GREEDY_BIAS": 0.50,
        "LOCAL_SEARCH_ITERS": 20,
        "SEED": None,
    }

    def __init__(self, instance_data, verbose: bool = False, params: Optional[Dict] = None,
                 initial_solution: Optional[Solution] = None):
        self.instance_data = instance_data
        self.verbose = verbose
        self.initial_solution = initial_solution

        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.POP_SIZE = int(self.params["POP_SIZE"])
        self.GENERATIONS = int(self.params["GENERATIONS"])
        self.CROSSOVER_RATE = float(self.params["CROSSOVER_RATE"])
        self.MUTATION_RATE = float(self.params["MUTATION_RATE"])
        self.TOURNAMENT_SIZE = int(self.params["TOURNAMENT_SIZE"])
        self.ELITISM = int(self.params["ELITISM"])
        self.TIME_LIMIT = float(self.params["TIME_LIMIT"])
        self.RANDOM_TOP_K = int(self.params.get("RANDOM_TOP_K", 5))
        self.GREEDY_BIAS = float(self.params.get("GREEDY_BIAS", 0.70))
        self.LOCAL_SEARCH_ITERS = int(self.params.get("LOCAL_SEARCH_ITERS", 20))

        seed = self.params.get("SEED")
        if seed is None:
            seed = time.time_ns()

        self.rng = random.Random(seed)

        self.base_scheduler = GreedyLookaheadScheduler(
            instance_data=instance_data,
            lookahead_limit=int(self.params.get("LOOKAHEAD_LIMIT", 4)),
            density_percentile=int(self.params.get("DENSITY_PERCENTILE", 25)),
            verbose=False,
        )

    def _score(self, schedule: List[Schedule]) -> int:
        return int(sum(s.fitness for s in schedule))

    def _solution(self, schedule: List[Schedule]) -> Solution:
        schedule = sorted(schedule, key=lambda s: (s.start, s.end, s.channel_id))
        return Solution(schedule, self._score(schedule))

    def _genre_of(self, sched: Schedule) -> str:
        info = self.base_scheduler.prog_by_id.get(sched.unique_program_id)
        return info[0].genre if info else ""

    def _state_from_schedule(self, schedule: List[Schedule]):
        if not schedule:
            return self.instance_data.opening_time, None, "", 0, set()

        schedule = sorted(schedule, key=lambda s: (s.start, s.end))
        last = schedule[-1]
        prev_ch = last.channel_id
        prev_genre = self._genre_of(last)
        streak = 0
        for s in reversed(schedule):
            if self._genre_of(s) == prev_genre:
                streak += 1
            else:
                break
        used = {s.unique_program_id for s in schedule}
        return last.end, prev_ch, prev_genre, streak, used

    def _make_schedule_item(self, candidate) -> Schedule:
        seg_score, _ch_idx, ch_id, prog, seg_start, seg_end = candidate
        return Schedule(
            program_id=prog.program_id,
            channel_id=ch_id,
            start=seg_start,
            end=seg_end,
            fitness=seg_score,
            unique_program_id=prog.unique_id,
        )

    def _pick_candidate(self, candidates):
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

        top_k = min(len(candidates), 10)
        pool = candidates[:top_k]

        # 100% random among top candidates
        return self.rng.choice(pool)

    def _stochastic_fill(self, prefix: Optional[List[Schedule]] = None) -> Solution:
        schedule = deepcopy(prefix) if prefix else []
        time_now, prev_ch, prev_genre, streak, used = self._state_from_schedule(schedule)

        while time_now < self.instance_data.closing_time:
            candidates = self.base_scheduler._get_candidates(time_now, prev_ch, prev_genre, streak, used)

            if not candidates:
                idx = bisect.bisect_right(self.base_scheduler.times, time_now)
                if idx < len(self.base_scheduler.times):
                    time_now = self.base_scheduler.times[idx]
                    continue
                break

            if len(candidates) == 1:
                cand = candidates[0]
            else:
                cand = self._pick_candidate(candidates)
                
            item = self._make_schedule_item(cand)
            _seg_score, _ch_idx, ch_id, prog, _seg_start, seg_end = cand

            schedule.append(item)
            used.add(prog.unique_id)

            if prog.genre == prev_genre:
                streak += 1
            else:
                prev_genre = prog.genre
                streak = 1

            prev_ch = ch_id
            time_now = seg_end

        return self._solution(schedule)

    def _init_population(self) -> List[Solution]:
        base = deepcopy(self.initial_solution) if self.initial_solution else self.base_scheduler.generate_solution()
        population = [base]

        while len(population) < self.POP_SIZE:
            # force more randomness for initial population
            old_bias = self.GREEDY_BIAS
            self.GREEDY_BIAS = 0.20
            population.append(self._stochastic_fill())
            self.GREEDY_BIAS = old_bias

        if self.verbose:
            print("Initial population scores:", [p.total_score for p in population])

        return population

    def _fitness(self, sol: Solution) -> int:
        return int(sol.total_score)

    def _select(self, population: List[Solution]) -> Solution:
        k = min(self.TOURNAMENT_SIZE, len(population))
        selected = self.rng.sample(population, k)
        return max(selected, key=self._fitness)

    def _repair_and_refill(self, raw_schedule: List[Schedule]) -> Solution:
        # Keep non-overlapping items in chronological order, then refill from the last valid point.
        raw_schedule = sorted(raw_schedule, key=lambda s: (s.start, -s.fitness))
        valid: List[Schedule] = []
        used = set()
        last_end = self.instance_data.opening_time

        for s in raw_schedule:
            if s.unique_program_id in used:
                continue
            if s.start < last_end:
                continue
            if s.end <= s.start:
                continue
            valid.append(deepcopy(s))
            used.add(s.unique_program_id)
            last_end = s.end

        return self._stochastic_fill(valid)

    def _crossover(self, p1: Solution, p2: Solution) -> Solution:
        if self.rng.random() > self.CROSSOVER_RATE:
            return deepcopy(max([p1, p2], key=self._fitness))

        opening = self.instance_data.opening_time
        closing = self.instance_data.closing_time
        split = self.rng.randint(opening + 1, closing - 1) if closing - opening > 2 else opening

        child_sched = []
        for s in p1.scheduled_programs:
            if s.end <= split:
                child_sched.append(deepcopy(s))
        used = {s.unique_program_id for s in child_sched}
        for s in p2.scheduled_programs:
            if s.start >= split and s.unique_program_id not in used:
                child_sched.append(deepcopy(s))
                used.add(s.unique_program_id)

        return self._repair_and_refill(child_sched)

    def _mutate(self, sol: Solution) -> Solution:
        if self.rng.random() > self.MUTATION_RATE or not sol.scheduled_programs:
            return sol

        schedule = sorted(sol.scheduled_programs, key=lambda s: (s.start, s.end))

        # Cut at a random gene and regenerate the suffix. This creates valid, meaningful variation.
        cut = self.rng.randint(0, len(schedule) - 1)
        prefix = deepcopy(schedule[:cut])
        mutated = self._stochastic_fill(prefix)

        # Keep mutation only if it is not catastrophically worse; this prevents drops like 1555 -> 747.
        if mutated.total_score >= sol.total_score * 0.85:
            return mutated
        return sol

    def _local_improve(self, sol: Solution) -> Solution:
        if self.LOCAL_SEARCH_ITERS <= 0 or not sol.scheduled_programs:
            return sol

        best = sol
        n = len(sol.scheduled_programs)
        for _ in range(min(self.LOCAL_SEARCH_ITERS, n)):
            if len(best.scheduled_programs) <= 1:
                break
            cut = self.rng.randint(0, len(best.scheduled_programs) - 1)
            candidate = self._stochastic_fill(best.scheduled_programs[:cut])
            if candidate.total_score > best.total_score:
                best = candidate
        return best

    def generate_random_solution(self) -> Solution:
        """
        Generate one stochastic/random solution.
        Used for experiments when you want different scores every run.
        """
        return self._stochastic_fill()

    def generate_solution(self) -> Solution:
        start = time.time()
        population = self._init_population()
        best = max(population, key=self._fitness)

        if self.verbose:
            print(f"Initial best: {best.total_score}, avg: {sum(p.total_score for p in population)/len(population):.1f}")

        for gen in range(1, self.GENERATIONS + 1):
            if time.time() - start >= self.TIME_LIMIT:
                break

            population = sorted(population, key=self._fitness, reverse=True)
            new_population = [deepcopy(ind) for ind in population[: self.ELITISM]]

            while len(new_population) < self.POP_SIZE:
                parent1 = self._select(population)
                parent2 = self._select(population)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            gen_best = max(population, key=self._fitness)
            if gen_best.total_score > best.total_score:
                best = deepcopy(gen_best)

            if self.verbose:
                avg = sum(p.total_score for p in population) / len(population)
                print(f"Gen {gen:3d}/{self.GENERATIONS} | Best: {best.total_score:5d} | Avg: {avg:7.1f}")

        best = self._local_improve(best)
        return best
