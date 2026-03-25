from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.greedy_lookahead_scheduler import GreedyLookaheadScheduler
from utils.utils import Utils
import argparse


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file", help="Path to input JSON (optional)")
    parser_arg.add_argument("--scheduler", "-s", dest="scheduler",
                            choices=["1", "2"],
                            help="Scheduler to use: 1=Beam, 2=GreedyLookahead")

    args = parser_arg.parse_args()

    file_path = select_file()
    parser = Parser(file_path)
    instance = parser.parse()
    Utils.set_current_instance(instance)

    print("\nOpening time:", instance.opening_time)
    print("Closing time:", instance.closing_time)
    print(f"Total Channels: {len(instance.channels)}")

    print('\nChoose scheduler:')
    print('1: Beam Search')
    print('2: Greedy + Lookahead')

    choice = args.scheduler if args.scheduler else input('Select scheduler [1/2] (default 1): ').strip() or '1'

    # Default parameters
    beam_width = 100
    lookahead = 4
    percentile = 25

    if choice == '2':
        print("\nRunning Greedy + Lookahead Scheduler...")
        scheduler = GreedyLookaheadScheduler(
            instance_data=instance,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False
        )
    else:
        print("\nRunning Beam Search Scheduler...")
        scheduler = BeamSearchScheduler(
            instance_data=instance,
            beam_width=beam_width,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False
        )

    solution = scheduler.generate_solution()

    print(f"\n Generated solution with total score: {solution.total_score}")

    algorithm_name = type(scheduler).__name__.lower()
    serializer = SolutionSerializer(input_file_path=file_path, algorithm_name=algorithm_name)
    serializer.serialize(solution)

    print(f"✓ Solution saved to output file")


if __name__ == "__main__":
    main()