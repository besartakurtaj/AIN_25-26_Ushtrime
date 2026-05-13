import random
import time
from copy import deepcopy
from typing import Dict, List, Optional

from models.solution import Solution
from models.schedule import Schedule
from scheduler.greedy_lookahead_scheduler import GreedyLookaheadScheduler


class LocalSearchScheduler:
    DEFAULT_PARAMS = {
        "MAX_ITERATIONS": 500,
        "RESTART_ITERATIONS": 50,
        "MAX_NO_IMPROVE": 100,
        "TIME_LIMIT": 300,
        "NEIGHBORHOOD_OPS": ["swap", "relocate", "2opt", "segment_reshuffle"],
        "FIRST_IMPROVEMENT": True,
        "RANDOM_RESTARTS": 3,
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

        self.MAX_ITERATIONS = int(self.params["MAX_ITERATIONS"])
        self.RESTART_ITERATIONS = int(self.params["RESTART_ITERATIONS"])
        self.MAX_NO_IMPROVE = int(self.params["MAX_NO_IMPROVE"])
        self.TIME_LIMIT = float(self.params["TIME_LIMIT"])
        self.FIRST_IMPROVEMENT = bool(self.params["FIRST_IMPROVEMENT"])
        self.RANDOM_RESTARTS = int(self.params["RANDOM_RESTARTS"])
        self.NEIGHBORHOOD_OPS = self.params.get("NEIGHBORHOOD_OPS", ["swap", "relocate", "2opt", "segment_reshuffle"])

        seed = self.params.get("SEED")
        if seed is None:
            seed = time.time_ns()

        self.rng = random.Random(seed)

        self.base_scheduler = GreedyLookaheadScheduler(
            instance_data=instance_data,
            lookahead_limit=4,
            density_percentile=25,
            verbose=False,
        )

        # Build mappings for efficiency
        self._build_mappings()

    def _build_mappings(self):
        """Build helper mappings for quick lookups."""
        self.prog_by_id = {}
        self.channels = set()
        
        for ch_data in self.instance_data.channels:
            self.channels.add(ch_data.channel_id)
            for prog in ch_data.programs:
                self.prog_by_id[prog.unique_id] = prog

    def _score(self, schedule: List[Schedule]) -> int:
        """Calculate total fitness score."""
        return int(sum(s.fitness for s in schedule))

    def _solution(self, schedule: List[Schedule]) -> Solution:
        """Create a Solution object from a schedule."""
        schedule = sorted(schedule, key=lambda s: (s.start, s.end, s.channel_id))
        return Solution(schedule, self._score(schedule))

    def _is_valid(self, schedule: List[Schedule]) -> bool:
        """Check if schedule is valid (no overlaps on same channel)."""
        by_channel = {}
        for s in schedule:
            if s.channel_id not in by_channel:
                by_channel[s.channel_id] = []
            by_channel[s.channel_id].append(s)
        
        for ch_id, items in by_channel.items():
            items_sorted = sorted(items, key=lambda x: x.start)
            for i in range(len(items_sorted) - 1):
                if items_sorted[i].end > items_sorted[i + 1].start:
                    return False
        
        return True

    def _swap_operator(self, schedule: List[Schedule]) -> Optional[Solution]:
        """
        Swap two programs between different positions/channels.
        Tries to move a low-fitness program to a better location.
        """
        if len(schedule) < 2:
            return None
        
        # Find the program with lowest fitness (excluding very high fitness ones)
        candidates = sorted(enumerate(schedule), key=lambda x: x[1].fitness)
        
        for idx, item in candidates[:min(5, len(candidates))]:
            # Try to insert this item at different positions
            for new_pos in range(len(schedule)):
                if abs(new_pos - idx) <= 1:
                    continue
                    
                new_schedule = deepcopy(schedule)
                moved_item = new_schedule.pop(idx)
                new_schedule.insert(new_pos, moved_item)
                
                if self._is_valid(new_schedule):
                    sol = self._solution(new_schedule)
                    if sol.total_score > self._score(schedule):
                        return sol
        
        return None

    def _relocate_operator(self, schedule: List[Schedule]) -> Optional[Solution]:
        """
        Move a program to a different time slot while keeping other programs.
        Effective for rearranging scheduling patterns.
        """
        if len(schedule) < 2:
            return None
        
        best_new_sol = None
        best_score = self._score(schedule)
        
        for idx in range(len(schedule)):
            item = schedule[idx]
            
            # Try moving to different positions
            for new_pos in range(max(0, idx - 5), min(len(schedule), idx + 6)):
                if new_pos == idx:
                    continue
                
                new_schedule = deepcopy(schedule)
                moved = new_schedule.pop(idx)
                new_schedule.insert(new_pos, moved)
                
                if self._is_valid(new_schedule):
                    sol = self._solution(new_schedule)
                    if sol.total_score > best_score:
                        best_score = sol.total_score
                        best_new_sol = sol
                        if self.FIRST_IMPROVEMENT:
                            return best_new_sol
        
        return best_new_sol

    def _2opt_operator(self, schedule: List[Schedule]) -> Optional[Solution]:
        """
        2-opt local search: reverse segments of the schedule.
        Useful for improving ordering within channels.
        """
        if len(schedule) < 3:
            return None
        
        best_new_sol = None
        best_score = self._score(schedule)
        
        for i in range(len(schedule) - 1):
            for j in range(i + 2, min(len(schedule), i + 8)):
                new_schedule = deepcopy(schedule)
                # Reverse the segment between i and j
                new_schedule[i:j+1] = reversed(new_schedule[i:j+1])
                
                if self._is_valid(new_schedule):
                    sol = self._solution(new_schedule)
                    if sol.total_score > best_score:
                        best_score = sol.total_score
                        best_new_sol = sol
                        if self.FIRST_IMPROVEMENT:
                            return best_new_sol
        
        return best_new_sol

    def _segment_reshuffle_operator(self, schedule: List[Schedule]) -> Optional[Solution]:
        """
        Reshuffle a segment by removing and reinserting programs.
        Helps escape local optima by creating larger changes.
        """
        if len(schedule) < 3:
            return None
        
        segment_size = min(4, max(2, len(schedule) // 4))
        start_idx = self.rng.randint(0, len(schedule) - segment_size)
        end_idx = start_idx + segment_size
        
        segment = deepcopy(schedule[start_idx:end_idx])
        self.rng.shuffle(segment)
        
        new_schedule = schedule[:start_idx] + segment + schedule[end_idx:]
        
        if self._is_valid(new_schedule):
            sol = self._solution(new_schedule)
            if sol.total_score > self._score(schedule):
                return sol
        
        return None

    def _apply_neighborhood_operator(self, schedule: List[Schedule]) -> Optional[Solution]:
        """Apply a random neighborhood operator."""
        op = self.rng.choice(self.NEIGHBORHOOD_OPS)
        
        if op == "swap":
            return self._swap_operator(schedule)
        elif op == "relocate":
            return self._relocate_operator(schedule)
        elif op == "2opt":
            return self._2opt_operator(schedule)
        elif op == "segment_reshuffle":
            return self._segment_reshuffle_operator(schedule)
        
        return None

    def _local_search_iteration(self, initial_solution: Solution, start_time: float) -> Solution:
        """Run one iteration of local search from initial solution."""
        current = initial_solution
        best = current
        no_improve_count = 0
        iteration = 0
        
        while iteration < self.MAX_ITERATIONS and no_improve_count < self.MAX_NO_IMPROVE:
            if time.time() - start_time >= self.TIME_LIMIT:
                break
            
            # Try neighborhood operators
            improved = False
            for _ in range(3):  # Try up to 3 different operators per iteration
                neighbor = self._apply_neighborhood_operator(current.scheduled_programs)
                
                if neighbor and neighbor.total_score > current.total_score:
                    current = neighbor
                    if current.total_score > best.total_score:
                        best = current
                        no_improve_count = 0
                    improved = True
                    if self.FIRST_IMPROVEMENT:
                        break
            
            if not improved:
                no_improve_count += 1
            
            iteration += 1
        
        if self.verbose:
            print(f"  LS iteration completed: initial={initial_solution.total_score}, "
                  f"final={best.total_score}, iterations={iteration}")
        
        return best

    def generate_solution(self, start_time: float = None) -> Solution:
        """
        Generate an optimized solution using Local Search with random restarts.
        
        Returns the best solution found across all restarts.
        """
        if start_time is None:
            start_time = time.time()
        
        # Start with greedy solution
        base_solution = self.initial_solution or self.base_scheduler.generate_solution()
        best_overall = base_solution
        
        if self.verbose:
            print(f"Starting LS from initial solution: {best_overall.total_score}")
        
        # Run local search with random restarts
        for restart_num in range(self.RANDOM_RESTARTS):
            if time.time() - start_time >= self.TIME_LIMIT:
                break
            
            if restart_num == 0:
                # First run: use the initial solution
                current = base_solution
            else:
                # Random restart: regenerate from greedy with some randomness
                current = self.base_scheduler.generate_solution()
            
            result = self._local_search_iteration(current, start_time)
            
            if result.total_score > best_overall.total_score:
                best_overall = result
            
            if self.verbose:
                print(f"Restart {restart_num + 1}/{self.RANDOM_RESTARTS}: best={best_overall.total_score}")
        
        return best_overall
