from typing import Set
import bisect

from models.solution import Solution
from models.schedule import Schedule
from models.program import Program

from scheduler.beam_search_scheduler import BeamSearchScheduler


class GreedyLookaheadScheduler(BeamSearchScheduler):
    """
    Improved Greedy Scheduler with Lookahead Simulation and local search.
    """

    def _simulate(self, time, prev_ch, prev_genre, g_streak, used: Set[str], depth: int):
        """
        Simulate future greedily for a limited depth with in-place set updates.
        """
        total_score = 0
        closing = self.instance_data.closing_time
        stack_used = []

        for _ in range(depth):
            if time >= closing:
                break

            candidates = self._get_candidates(time, prev_ch, prev_genre, g_streak, used)
            if not candidates:
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times):
                    time = self.times[idx]
                    continue
                else:
                    break

            # Improved scoring heuristic
            def score_candidate(x):
                seg_score, ch_idx, ch_id, prog, seg_start, seg_end = x
                streak_penalty = 0.9 if prog.genre == prev_genre and g_streak >= 2 else 1.0
                duration_factor = (seg_end - seg_start) / 60  # favor longer segments
                future_time_bonus = (closing - seg_end) * self.avg_score_per_min
                availability_factor = 1.0
                return seg_score * streak_penalty * availability_factor + future_time_bonus * duration_factor

            candidates.sort(key=score_candidate, reverse=True)
            seg_score, ch_idx, ch_id, prog, seg_start, seg_end = candidates[0]

            total_score += seg_score
            used.add(prog.unique_id)
            stack_used.append(prog.unique_id)

            # Update streak and genre
            if prog.genre == prev_genre:
                g_streak += 1
            else:
                g_streak = 1
                prev_genre = prog.genre

            prev_ch = ch_id
            time = seg_end

        # backtrack used
        for uid in stack_used:
            used.remove(uid)

        return total_score

    def _greedy_lookahead(self):
        """
        Main greedy + lookahead algorithm with improved heuristics.
        """
        opening = self.instance_data.opening_time
        closing = self.instance_data.closing_time

        time = opening
        prev_ch = None
        prev_genre = ""
        g_streak = 0

        schedule = []
        total_score = 0
        used = set()

        while time < closing:
            candidates = self._get_candidates(time, prev_ch, prev_genre, g_streak, used)

            if not candidates:
                idx = bisect.bisect_right(self.times, time)
                if idx < len(self.times):
                    time = self.times[idx]
                    continue
                else:
                    break

            # Threshold-based candidate pruning
            max_score = max(c[0] for c in candidates)
            candidates = [c for c in candidates if c[0] >= 0.8 * max_score]

            best_candidate = None
            best_value = float('-inf')

            for cand in candidates:
                seg_score, ch_idx, ch_id, prog, seg_start, seg_end = cand
                new_streak = 1 if prog.genre != prev_genre else g_streak + 1

                # Dynamic lookahead depth
                depth = min(self.lookahead_limit, max(2, int((closing - seg_end) / 60)))

                # simulate future
                future_score = self._simulate(
                    seg_end,
                    ch_id,
                    prog.genre,
                    new_streak,
                    used | {prog.unique_id},
                    depth
                )

                # optional 2-step lookahead with reduced weight
                future_score_2 = 0
                if depth >= 3:
                    future_score_2 = 0.5 * self._simulate(
                        seg_end + 1,  # small offset
                        ch_id,
                        prog.genre,
                        new_streak,
                        used | {prog.unique_id},
                        2
                    )

                total_estimated = seg_score + future_score + future_score_2

                if total_estimated > best_value:
                    best_value = total_estimated
                    best_candidate = cand

            # Apply best candidate
            seg_score, ch_idx, ch_id, prog, seg_start, seg_end = best_candidate

            schedule.append(Schedule(
                program_id=prog.program_id,
                channel_id=ch_id,
                start=seg_start,
                end=seg_end,
                fitness=seg_score,
                unique_program_id=prog.unique_id
            ))

            total_score += seg_score
            used.add(prog.unique_id)

            if prog.genre == prev_genre:
                g_streak += 1
            else:
                g_streak = 1
                prev_genre = prog.genre

            prev_ch = ch_id
            time = seg_end

        return Solution(schedule, total_score)

    def generate_solution(self) -> Solution:
        """
        Public method to generate solution.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("GREEDY + LOOKAHEAD SCHEDULER (IMPROVED)")
            print("="*70)

        sol = self._greedy_lookahead()

        # Adaptive local search iterations
        iter_limit = 30 if self.n_channels <= 50 else 15
        sol = self._local_search(sol, max_iter=iter_limit)

        if self.verbose:
            print(f"Score: {sol.total_score}")
            print(f"Programs: {len(sol.scheduled_programs)}")
            print("="*70 + "\n")

        return sol