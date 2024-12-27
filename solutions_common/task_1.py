import argparse
import time

from lib.api.client import QualifiersApiClient as ApiClient
from lib.robot.basic_controller import BasicRobotController
from lib.robot.feedback_controller import FeedbackRobotController
from lib.robot.sensors_reader import SimpleSensorReader
from lib.maze.explore.algo_base import RefinementPolicy
from lib.maze.explore.tremaux import TremauxExplorer
from lib.maze.explore.right_hand import RightHandExplorer
from lib.maze.explore.heuristic import (FirstEntityHeuristic,
                                        RandomHeuristic,
                                        ManhattanHeuristic)


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--api_token', type=str,
                        help='Your API token', default="Lorem ipsum dolor sit amet")

    args = parser.parse_args()
    return args


def main(api_token: str):
    start_time = time.monotonic()

    client = ApiClient(api_token,
                       max_retries=10,
                       timeout=5.,
                       sleep_time=0.05)
    sensors_reader = SimpleSensorReader(api_client=client)

    # Option 1: Simple controller
    # controller = BasicRobotController(api_client=client, 
    #                                   delay=0.1)

    # Option 2: Feedback controller
    controller = FeedbackRobotController(api_client=client,
                                         sensors_reader=sensors_reader,
                                         delay=0.010,
                                         api_move_wait_timeout=5.0,
                                         api_command_retries=1)

    # Option 1: Tremaux
    explorer = TremauxExplorer(controller=controller,
                                     sensors_reader=sensors_reader,
                                     walls_threshold=65.,
                                     yaw_eps=15.,
                                     heuristic=RandomHeuristic(),
                                     stop_at_center=False,
                                     refine_maze=RefinementPolicy.EXACT,
                                     semifinal_mode=False)

    # Option 2: Right Hand

    # explorer = RightHandExplorer(controller=controller,
    #                              sensors_reader=sensors_reader)

    maze, _ = explorer.run()
    score = client.send_maze(maze)

    print(maze)
    print(f"\nScore: {score}, time {time.monotonic()-start_time}")


if __name__ == "__main__":
    args = _parse_arguments()
    if not args.api_token:
        print("api_token key required for this task!")
    else:
        main(args.api_token)
