import argparse
import numpy as np

from lib.api.client import QualifiersApiClient as ApiClient
from lib.robot.basic_controller import BasicRobotController
from lib.robot.feedback_controller import FeedbackRobotController
from lib.robot.sensors_reader import SimpleSensorReader
from lib.maze.explore.algo_base import RefinementPolicy
from lib.maze.explore.tremaux import TremauxExplorer
from lib.maze.explore.flood_fill import FloodFillExplorer
from lib.maze.explore.right_hand import RightHandExplorer
from lib.maze.explore.heuristic import (FirstEntityHeuristic,
                                        RandomHeuristic,
                                        ManhattanHeuristic)
from lib.maze.plan.bfs import BFSPlanner


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--api_token', type=str,
                        help='Your API token', default="Lorem ipsum dolor sit amet")

    args = parser.parse_args()
    return args


def main(api_token: str):
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

    # Try 1: Exploration
    # Option 1: TremauxExplorer + BFS Planner
    # explorer = TremauxExplorer(controller=controller,
    #                                  sensors_reader=sensors_reader,
    #                                  walls_threshold=65.,
    #                                  yaw_eps=15.,
    #                                  heuristic=ManhattanHeuristic(
    #                                      target_point=np.array([7.5, 7.5])
    #                                  ),
    #                                  stop_at_center=False,
    #                                  refine_maze=RefinementPolicy.APPROX,
    #                                  semifinal_mode=False)
    # 
    
    # Option 2: FloodFillExplorer (explorer + planner)
    explorer = FloodFillExplorer(controller=controller,
                                 sensors_reader=sensors_reader,
                                 walls_threshold=65.,
                                 yaw_eps=15.,
                                 semifinal_mode=False)
    
    maze, misc_dict = explorer.run()
    maze, _ = explorer.run()
    
    client.reset_maze()

    # Try 2: Path planning and navigation
    # Option 1: Simple controller
    # controller = BasicRobotController(api_client=client,
    #                                   delay=0.1)

    # Option 2: Feedback controller
    controller = FeedbackRobotController(api_client=client,
                                         sensors_reader=sensors_reader,
                                         delay=0.010,
                                         api_move_wait_timeout=5.0,
                                         api_command_retries=1)

    # Option 1: BFS Planner (for TremauxExplorer)
    # planner = BFSPlanner(controller, sensors_reader, maze)
    # plan = planner.run()

    # Option 2: FloodFillExplorer (explorer + planner)
    planner = FloodFillExplorer(controller=controller,
                                 sensors_reader=sensors_reader,
                                 walls_threshold=65.,
                                 yaw_eps=15.,
                                 prebuild_walls=misc_dict['walls'],
                                 semifinal_mode=False)
    maze, misc_dict = planner.run()
    
    client.reset_maze()

    # Try 3: pedal to the metal
    # Option 1: Simple controller
    # controller = BasicRobotController(api_client=client,
    #                                   delay=0.1)
    # Option 2: Feedback controller
    controller = FeedbackRobotController(api_client=client,
                                         sensors_reader=sensors_reader,
                                         delay=0.010,
                                         api_move_wait_timeout=5.0,
                                         api_command_retries=1)

    # Option 1: BFS Planner (for TremauxExplorer) with calculated plan
    # planner = BFSPlanner(controller, sensors_reader, maze)
    # _ = planner.run(precalculated_plan=plan)
    
    # Option 2: FloodFillExplorer (explorer + planner)
    planner = FloodFillExplorer(controller=controller,
                                 sensors_reader=sensors_reader,
                                 walls_threshold=65.,
                                 yaw_eps=15.,
                                 prebuild_walls=misc_dict['walls'],
                                 semifinal_mode=False)
    _, _ = planner.run()




if __name__ == "__main__":
    args = _parse_arguments()
    if not args.api_token:
        print("api_token key required for this task!")
    else:
        main(args.api_token)
