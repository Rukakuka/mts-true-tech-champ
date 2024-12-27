import argparse
import time
from datetime import datetime

from lib.api.client import SemifinalApiClient
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
    parser.add_argument('--id', type=str, help='Your ID')
    parser.add_argument('--ip', type=str, help='Your robot IP')
    args = parser.parse_args()
    return args


def main(id: str, ip: str):
    start_time = time.monotonic()

    client = SemifinalApiClient(id=id, robot_ip=ip,
                                max_retries=10,
                                timeout=5.,
                                sleep_time=0.05)
    sensors_reader = SimpleSensorReader(api_client=client)

    now = datetime.now().isoformat()
    with open(f'log{str(now)}.log', 'w', encoding="utf-8") as logfile:
        for i in range(9):
            reading = sensors_reader.get_reading()
            logfile.writelines(str(reading) + '\n')
                
    print(f"10 readings took {time.monotonic() - start_time} s, with average {(time.monotonic() - start_time)/10} s per measure and {10/(time.monotonic() - start_time)} hz rate")


if __name__ == "__main__":
    args = _parse_arguments()
    if not args.id:
        print("'id' key required for this task!")
    if not args.ip:
        print("'ip' key required for this task!")
    else:
        main(args.id, args.ip)
