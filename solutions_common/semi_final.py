import argparse

from lib.api.client import SemifinalApiClient
from lib.robot.basic_controller import SemifinalBasicRobotController
from lib.maze.explore.flood_fill import FloodFillExplorer
from lib.robot.sensors_reader import SimpleSensorReaderSemiFinal, SimpleSensorReaderSemiFinalAsync
from lib.api.initialization import wait_api_init
from lib.robot.simple_logger import SimpleLogger

def _parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--robot_id', type=str, help='Robot ID')
    parser.add_argument('--robot_ip', type=str, help='Robot IP')

    args = parser.parse_args()
    return args


def main(robot_id, robot_ip):
    client = SemifinalApiClient(id=robot_id,
                                robot_ip=robot_ip,
                                max_retries=10,
                                timeout=5,
                                sleep_time=0.05)
    
    logger = SimpleLogger()
    
    # sensors_reader = SimpleSensorReaderSemiFinal(api_client=client)
    sensors_reader = SimpleSensorReaderSemiFinalAsync(api_client=client, logger=logger)
    sensors_reader.start()

    wait_api_init(sensors_reader=sensors_reader)
    
    client.update_yaw_offset()

    controller = SemifinalBasicRobotController(api_client=client,
                                               delay=0.01)

    explorer = FloodFillExplorer(controller=controller,
                                 sensors_reader=sensors_reader,
                                 walls_threshold=101.,
                                 yaw_eps=35.,
                                 semifinal_mode=True)

    explorer.run()
    sensors_reader.stop()
    logger.close()
    print("Done!")


if __name__ == "__main__":
    args = _parse_arguments()
    if not args.robot_id or not args.robot_ip:
        print("Robot ID and IP are required!")
    else:
        main(args.robot_id, args.robot_ip)
