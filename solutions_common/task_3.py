import argparse
import numpy as np

from lib.api.client import QualifiersApiClient as ApiClient
from lib.robot.pd_pwm_controller import PdPwmRobotController
from lib.robot.async_poll_pid_controller import AsyncPollPidController
from lib.robot.discrete_async_poll_controller import DiscreteAsyncPollController
from lib.robot.sensors_reader import SimpleSensorReader
from lib.maze.explore.tremaux import TremauxExplorer
from lib.maze.explore.heuristic import (FirstEntityHeuristic, 
                                        RandomHeuristic, 
                                        ManhattanHeuristic)
from lib.maze.explore.flood_fill import FloodFillExplorer
from lib.maze.plan.bfs import BFSPlanner
from lib.maze.explore.algo_base import RefinementPolicy


def _parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--api_token', type=str, help='Your API token', default="Lorem ipsum dolor sit amet")

    args = parser.parse_args()
    return args


def main(api_token: str):
    client = ApiClient(api_token,
                       max_retries=10,
                       timeout=10,
                       sleep_time=0.05)

    # Try 1: Exploration
    controller = PdPwmRobotController(api_client=client, 
                                      delay=0.1,
                                      kp_linear=15,    
                                      kd_linear=5.3,
                                      kp_angular=70, kd_angular=6,
                                      distance_offset=5.0,
                                      angle_offset=0.2,
                                      success_rate_angular=2,
                                      success_rate_linear=3,
                                      lenght=1, 
                                      radius=1, 
                                      pwm_max = 255, 
                                      pwm_time=0.09)

    sensors_reader = SimpleSensorReader(api_client=client)
    explorer = TremauxExplorer(controller=controller,
                                     sensors_reader=sensors_reader,
                                     walls_threshold=100.,
                                     yaw_eps=15.,
                                     heuristic=RandomHeuristic(),
                                    #  heuristic=ManhattanHeuristic(
                                    #      target_point=np.array([7.5, 7.5])
                                    #  ),
                                     stop_at_center=True,
                                     refine_maze=RefinementPolicy.APPROX,
                                     semifinal_mode=False)
    maze, _ = explorer.run()
    client.reset_maze()

    # Try 2: Path planning and navigation
    controller = PdPwmRobotController(api_client=client, 
                                      delay=0.1,
                                      kp_linear=15,    
                                      kd_linear=5.3,
                                      kp_angular=70, kd_angular=6,
                                      distance_offset=5.0,
                                      angle_offset=0.2,
                                      success_rate_angular=2,
                                      success_rate_linear=3,
                                      lenght=1, 
                                      radius=1, 
                                      pwm_max = 255, 
                                      pwm_time=0.09)
    sensors_reader = SimpleSensorReader(api_client=client)
    planner = BFSPlanner(controller, sensors_reader, maze)
    plan = planner.run()
    client.reset_maze()

    # Try 3: Pedal to the metal
    controller = PdPwmRobotController(api_client=client, 
                                      delay=0.1,
                                      kp_linear=15,    
                                      kd_linear=5.3,
                                      kp_angular=70, kd_angular=6,
                                      distance_offset=5.0,
                                      angle_offset=0.2,
                                      success_rate_angular=2,
                                      success_rate_linear=3,
                                      lenght=1, 
                                      radius=1, 
                                      pwm_max = 255, 
                                      pwm_time=0.09)
    sensors_reader = SimpleSensorReader(api_client=client)
    planner = BFSPlanner(controller, sensors_reader, maze)
    _ = planner.run(precalculated_plan=plan)


if __name__ == "__main__":
    args = _parse_arguments()
    if not args.api_token:
        print("api_token key required for this task!")
    else:
        main(args.api_token)
