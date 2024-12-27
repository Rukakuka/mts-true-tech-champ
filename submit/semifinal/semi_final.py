from __future__ import annotations
import time
import requests
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import enum
from abc import ABC, abstractmethod
from collections import deque

ROBOT_IP = "127.0.0.1:8801"
ROBOT_ID = ""

API_TOKEN = ""
_UNVISITED = 0
_VISITED = 1
_BLOCKED = 2 * _VISITED
MAZE_SIDE = 16
CELL_TYPES = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 0, 1): 5,
    (0, 0, 1, 1): 6,
    (0, 1, 1, 0): 7,
    (1, 1, 0, 0): 8,
    (1, 0, 1, 0): 9,
    (0, 1, 0, 1): 10,
    (0, 1, 1, 1): 11,
    (1, 1, 1, 0): 12,
    (1, 1, 0, 1): 13,
    (1, 0, 1, 1): 14,
    (1, 1, 1, 1): 15,
}


class Orientation(enum.IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


WALLS_DICT = {
    0: {
        Orientation.WEST: False,
        Orientation.NORTH: False,
        Orientation.EAST: False,
        Orientation.SOUTH: False
    },
    1: {
        Orientation.WEST: True,
        Orientation.NORTH: False,
        Orientation.EAST: False,
        Orientation.SOUTH: False
    },
    2: {
        Orientation.WEST: False,
        Orientation.NORTH: True,
        Orientation.EAST: False,
        Orientation.SOUTH: False
    },
    3: {
        Orientation.WEST: False,
        Orientation.NORTH: False,
        Orientation.EAST: True,
        Orientation.SOUTH: False
    },
    4: {
        Orientation.WEST: False,
        Orientation.NORTH: False,
        Orientation.EAST: False,
        Orientation.SOUTH: True
    },
    5: {
        Orientation.WEST: True,
        Orientation.NORTH: False,
        Orientation.EAST: False,
        Orientation.SOUTH: True
    },
    6: {
        Orientation.WEST: False,
        Orientation.NORTH: False,
        Orientation.EAST: True,
        Orientation.SOUTH: True
    },
    7: {
        Orientation.WEST: False,
        Orientation.NORTH: True,
        Orientation.EAST: True,
        Orientation.SOUTH: False
    },
    8: {
        Orientation.WEST: True,
        Orientation.NORTH: True,
        Orientation.EAST: False,
        Orientation.SOUTH: False
    },
    9: {
        Orientation.WEST: True,
        Orientation.NORTH: False,
        Orientation.EAST: True,
        Orientation.SOUTH: False
    },
    10: {
        Orientation.WEST: False,
        Orientation.NORTH: True,
        Orientation.EAST: False,
        Orientation.SOUTH: True
    },
    11: {
        Orientation.WEST: False,
        Orientation.NORTH: True,
        Orientation.EAST: True,
        Orientation.SOUTH: True
    },
    12: {
        Orientation.WEST: True,
        Orientation.NORTH: True,
        Orientation.EAST: True,
        Orientation.SOUTH: False
    },
    13: {
        Orientation.WEST: True,
        Orientation.NORTH: True,
        Orientation.EAST: False,
        Orientation.SOUTH: True
    },
    14: {
        Orientation.WEST: True,
        Orientation.NORTH: False,
        Orientation.EAST: True,
        Orientation.SOUTH: True
    },
    15: {
        Orientation.WEST: True,
        Orientation.NORTH: True,
        Orientation.EAST: True,
        Orientation.SOUTH: True
    }
}


@dataclass
class SensorsReading:

    front_distance: float
    right_side_distance: float
    left_side_distance: float
    back_distance: float
    left_45_distance: float
    right_45_distance: float
    rotation_pitch: float
    rotation_yaw: float
    rotation_roll: float
    down_x_offset: float
    down_y_offset: float


class RefinementPolicy(enum.Enum):
    NO = "no"
    EXACT = "exact"
    APPROX = "approx"


class AbstractMazeExplorer(ABC):
    @abstractmethod
    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass


class Orientation(enum.IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class PossibleDirection(enum.IntEnum):
    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3


class AbstractSensorsReader(ABC):
    @abstractmethod
    def get_reading(self) -> SensorsReading:
        pass

    def start():
        pass

    def stop():
        pass


# class SimpleSensorReader(AbstractSensorsReader):
#     def __init__(self, api_client: ApiClient):
#         super(SimpleSensorReader, self).__init__()
#         self._api_client = api_client

#     def get_reading(self) -> SensorsReading:
#         return self._api_client.read_sensors()
def wrap_angle(angle):
    # Use modulo to wrap the angle
    wrapped = angle % 360

    # Handle negative angles
    if wrapped < 0:
        wrapped += 360

    return wrapped


@dataclass
class SemifinalSensorsReading(SensorsReading):

    def __init__(self, json, yaw_offset: float):
        self.back_distance = json["laser"]["1"]
        self.left_side_distance = json["laser"]["2"]
        self.right_45_distance = json["laser"]["3"]
        self.front_distance = json["laser"]["4"]
        self.right_side_distance = json["laser"]["5"]
        self.left_45_distance = json["laser"]["6"]
        self.rotation_roll = json["imu"]["roll"]
        self.rotation_pitch = json["imu"]["pitch"]
        self.rotation_yaw = wrap_angle(json["imu"]["yaw"] - yaw_offset)
        self.down_x_offset = 0
        self.down_y_offset = 0

    # def as_normal_values(self) -> SemifinalSensorsReading:
    #     SemifinalSensorsReading(self._laser_to_meters(self.front_distance),
    #                             self._laser_to_meters(
    #                                 self.right_side_distance),
    #                             self._laser_to_meters(self.left_side_distance),
    #                             self._laser_to_meters(self.back_distance),
    #                             self._laser_to_meters(self.left_45_distance),
    #                             self._laser_to_meters(self.right_45_distance),
    #                             self._imu_to_radians(self.rotation_pitch),
    #                             self._imu_to_radians(self.rotation_yaw),
    #                             self._imu_to_radians(self.rotation_roll),
    #                             0,
    #                             0)

    # def _imu_to_radians(self, value: float) -> float:
    #     return value * DEGREES_TO_RADIANS

    # def _laser_to_meters(self, value: float) -> float:
    #     return value * LASER_TO_METERS

    def is_zero(self) -> bool:
        return not np.any(np.array([self.front_distance,
                                    self.right_side_distance,
                                    self.left_side_distance,
                                    self.back_distance,
                                    self.left_45_distance,
                                    self.right_45_distance,
                                    self.rotation_pitch,
                                    self.rotation_yaw,
                                    self.rotation_roll,
                                    self.down_x_offset,
                                    self.down_y_offset]))


class SimpleSensorReaderSemiFinal(AbstractSensorsReader):
    """Simple synchronous implementation of the sensors reader.
    """

    def __init__(self, api_client: SemifinalApiClient):
        """Initializes the synchronous implementation of the sensors reader for semifinal.

        Args:
            api_client (ApiClient): API client instance.
        """
        super(SimpleSensorReaderSemiFinal, self).__init__()
        self._api_client = api_client

    def get_reading(self) -> SemifinalSensorsReading:
        return self._api_client.read_sensors()


class OdometryState:
    def __init__(self, cell: np.ndarray, orientation: Orientation):
        self._cell = cell
        self._orientation = orientation

    @property
    def cell(self) -> np.ndarray:
        return self._cell.copy()

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    def rotatate_right(self) -> OdometryState:
        value = self._orientation.value
        value = (value + 1) % 4
        new_orientation = Orientation(value)
        return OdometryState(self._cell.copy(), new_orientation)

    def rotate_left(self) -> OdometryState:
        value = self._orientation.value
        value = value - 1
        if value == -1:
            value = 3
        new_orientation = Orientation(value)
        return OdometryState(self._cell.copy(), new_orientation)

    def move_forward(self) -> OdometryState:
        return OdometryState(
            self._linear_movement(np.array([-1, 0])), self._orientation
        )

    def move_backward(self) -> OdometryState:
        return OdometryState(self._linear_movement(np.array([1, 0])), self._orientation)

    def _build_rotation_matrix(self) -> np.ndarray:
        orientation = self._orientation
        angle = 0.0
        if orientation == Orientation.NORTH:
            angle = 0.0
        elif orientation == Orientation.EAST:
            angle = -np.pi / 2.0
        elif orientation == Orientation.SOUTH:
            angle = -np.pi
        elif orientation == Orientation.WEST:
            angle = np.pi / 2.0
        else:
            pass
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return rotation_matrix

    def _linear_movement(self, displacement: np.ndarray) -> np.ndarray:
        rotation_matrix = self._build_rotation_matrix()
        current_cell = self._cell
        displacement = rotation_matrix @ displacement
        new_cell = current_cell + displacement.astype(int)
        return new_cell


class AbstractRobotController(ABC):
    @property
    @abstractmethod
    def odometry(self) -> OdometryState:
        pass

    @abstractmethod
    def move_forward(self, n: int = 1):
        pass

    @abstractmethod
    def move_backward(self, n: int = 1):
        pass

    @abstractmethod
    def rotate_left(self, n: int = 1):
        pass

    @abstractmethod
    def rotate_right(self, n: int = 1):
        pass


class SemifinalBasicRobotController(AbstractRobotController):

    _CELL2MM = 168  # We tried this also 130
    _TURN2DEG = 90

    def __init__(self, api_client: SemifinalApiClient, delay: Optional[float]):
        super(SemifinalBasicRobotController, self).__init__()
        self._api_client = api_client
        self._delay = delay
        self._odometry = OdometryState(np.array([15, 0]), Orientation.NORTH)

    @property
    def odometry(self) -> OdometryState:
        """Current cell position and orientation of the robot.
        """
        return self._odometry

    def move_forward(self, n: int = 1):
        """Moves robot n cells forward (in local frame).
        """
        self._api_client.forward(n * self._CELL2MM)
        self._odometry = self._odometry.move_forward()
        self._do_delay()

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        self._api_client.backward(n * self._CELL2MM)
        self._odometry = self._odometry.move_backward()
        self._do_delay()

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        self._api_client.left(n * self._TURN2DEG)
        self._odometry = self._odometry.rotate_left()
        self._do_delay()

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        self._api_client.right(n * self._TURN2DEG)
        self._odometry = self._odometry.rotatate_right()
        self._do_delay()

    def _do_delay(self):
        if self._delay is not None:
            time.sleep(self._delay)


class AbstractMazeExplorer(ABC):
    @abstractmethod
    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass


class AbstractJunctionHeuristic(ABC):
    @abstractmethod
    def select(
        self, directions: List[Tuple[PossibleDirection, np.ndarray]]
    ) -> Tuple[PossibleDirection, np.ndarray]:
        pass


class FirstEntityHeuristic(AbstractJunctionHeuristic):
    def select(
        self, directions: List[Tuple[PossibleDirection, np.ndarray]]
    ) -> Tuple[PossibleDirection, np.ndarray]:
        return directions[0]


class RandomHeuristic(AbstractJunctionHeuristic):
    def select(
        self, directions: List[Tuple[PossibleDirection, np.ndarray]]
    ) -> Tuple[PossibleDirection, np.ndarray]:

        if len(directions) == 0:
            return directions[0]
        indicies = np.arange(len(directions))
        idx = np.random.choice(indicies)
        return directions[idx]


class ManhattanHeuristic(AbstractJunctionHeuristic):
    def __init__(self, target_point: np.ndarray):
        super(ManhattanHeuristic, self).__init__()
        self._target_point = target_point.copy()

    def select(
        self, directions: List[Tuple[PossibleDirection, np.ndarray]]
    ) -> Tuple[PossibleDirection, np.ndarray]:
        if len(directions) == 0:
            return directions[0]
        values = [np.linalg.norm(e[1] - self._target_point, ord=1)
                  for e in directions]
        idx = np.argmin(values)
        return directions[idx]


class AbstractPlanner(ABC):
    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 maze: np.ndarray):
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._maze = maze.copy()
        self._center_entrance = self._find_center_entrance()

    @abstractmethod
    def find_path(self, start_cell: np.ndarray, target_cell: np.ndarray):
        pass

    def run(self, precalculated_plan=None):
        start_cell = self._controller.odometry.cell
        plan = self.find_path(
            start_cell, self._center_entrance) if precalculated_plan is None else precalculated_plan
        self._execute_plan(plan)
        return plan

    def _execute_plan(self, plan: np.ndarray):
        current_orientation = self._controller.odometry.orientation

        for i in range(1, plan.shape[0]):
            prev_cell, curr_cell = plan[i-1], plan[i]
            dx, dy = curr_cell[0] - prev_cell[0], curr_cell[1] - prev_cell[1]

            if dx == 1:
                target_orientation = Orientation.SOUTH
            elif dx == -1:
                target_orientation = Orientation.NORTH
            elif dy == 1:
                target_orientation = Orientation.EAST
            else:
                target_orientation = Orientation.WEST

            current_index = current_orientation.value
            target_index = target_orientation.value

            turn_difference = (target_index - current_index + 4) % 4

            if turn_difference == 0:
                pass
            elif turn_difference == 1:
                self._controller.rotate_right()
            elif turn_difference == 2:
                pass
            else:
                self._controller.rotate_left()

            current_orientation = target_orientation

            if turn_difference == 2:
                self._controller.move_backward()
            else:
                self._controller.move_forward()

    def _find_center_entrance(self) -> np.ndarray:
        central_cells = [(7, 7), (7, 8), (8, 7),  (8, 8)]
        dict_enter = {

            (7, 6): [0, 1, 2, 4, 5, 8, 10, 13],
            (8, 6): [0, 1, 2, 4, 5, 8, 10, 13],

            (6, 7): [0, 1, 2, 3, 7, 8, 9, 12],
            (6, 8): [0, 1, 2, 3, 7, 8, 9, 12],

            (9, 7): [0, 1, 3, 4, 5, 6, 9, 14],
            (9, 8): [0, 1, 3, 4, 5, 6, 9, 14],

            (7, 9): [0, 2, 3, 4, 6, 7, 10, 11],
            (8, 9): [0, 2, 3, 4, 6, 7, 10, 11],
        }
        for central_cell in central_cells:
            if self._is_entry(central_cell, dict_enter):
                return np.array(central_cell)

    def _is_entry(self,
                  cell_candidate: Tuple[int, int],
                  dict_enter: Dict[Tuple[int, int], List[int]]) -> bool:
        x, y = cell_candidate
        neighbours_list = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        neighbour_list_types = []
        for cell in neighbours_list:
            x, y = cell
            neighbour_list_types.append(self._maze[x, y])

        neighbours_list = np.array(neighbours_list)
        neighbour_list_types = np.array(neighbour_list_types)

        neighbours_list_filtered = neighbours_list[neighbour_list_types != -1]
        neighbours_list_filtered_tuples = [
            tuple(x) for x in neighbours_list_filtered]
        for neighbour in neighbours_list_filtered_tuples:
            x, y = neighbour
            value = self._maze[x, y,]
            if neighbour in dict_enter and value in dict_enter[neighbour]:
                return True


class BaseClient:
    """Basic class to make requests for the API.
    """

    def _request(self, request_type: str, url: str, data: dict = None) -> Optional[requests.Response]:
        """
            Makes a request to the API.
            :param request_type: The type of request to make.
            :param url: The URL to make the request to.
            :param data: The data to send with the request.
            :param max_retries: The maximum number of retries to make.
            :param timeout: The timeout for the request.
            :return: The response from the API.
        """
        for attempt in range(self._max_retries):
            try:
                if request_type == "get":
                    response = requests.get(url, timeout=self._timeout)
                elif request_type == "post":
                    response = requests.post(
                        url, timeout=self._timeout, json=data)
                elif request_type == "put":
                    response = requests.put(
                        url, timeout=self._timeout, json=data)
                else:
                    raise ValueError()
                response.raise_for_status()
                if response.status_code != 200:
                    print(
                        f"Received error status code {response.status_code}")
                else:
                    return response
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, requests.exceptions.RequestException) as err:
                print(f"Attempt {attempt + 1} failed: {err}")
                pass

            if attempt < self._max_retries - 1:
                print(f"Retrying ({attempt + 1}/{self._max_retries})")
                time.sleep(self._sleep_time)
            else:
                print("Max retries reached. Exiting.")

        return None


class SemifinalApiClient(BaseClient):
    def __init__(self, id: str, robot_ip: str, max_retries: int = 5, timeout: float = 5.0, sleep_time: float = 0.1) -> None:
        self._id = id
        self._robot_ip = robot_ip
        self._base_url = f"http://{robot_ip}"
        self._url_sensors = f"{self._base_url}/sensor"
        self._url_move = f"{self._base_url}/move"
        self._url_motor = f"{self._base_url}/motor"

        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep_time = sleep_time

        self._yaw_offset = 0

    def read_sensors(self) -> Optional[SemifinalSensorsReading]:
        """Reads sensors values from the REST API.
        """
        response = self._request(
            request_type="post",
            url=self._url_sensors,
            data={"id": self._id,
                  "type": "all"
                  }
        )
        return SemifinalSensorsReading(response.json(), self._yaw_offset) if response is not None else None

    def update_yaw_offset(self):
        reading = self.read_sensors()
        yaw = reading.rotation_yaw
        self._yaw_offset = yaw

    def forward(self, mm: int):
        self._movement_request("forward", mm)

    def backward(self, mm: int):
        self._movement_request("backward", mm)

    def left(self, degrees: int):
        return self._rotation_request("left", degrees)

    def right(self, degrees: int):
        return self._rotation_request("right", degrees)

    def pwm(self, pwm_l: int, pwm_l_dt: float, pwm_r: int, pwm_r_dt: float):
        response = self._request(
            request_type="put",
            url=self._url_motor,
            data={"id": self._id,
                  "l": int(np.clip(pwm_l, -255, 255)),
                  "r": int(np.clip(pwm_r, -255, 255)),
                  "l_time": int(np.clip(pwm_l_dt, 0, 10000)),
                  "r_time": int(np.clip(pwm_r_dt, 0, 10000))
                  }
        )
        return response.json() if response is not None else None

    def _rotation_request(self, direction: str, degrees: int):
        response = self._request(
            request_type="put",
            url=self._url_move,
            data={"id": self._id,
                  "direction": direction,
                  "len": int(np.clip(degrees, 0, 360))
                  }
        )

    def _movement_request(self, direction: str, mm: int):
        response = self._request(
            request_type="put",
            url=self._url_move,
            data={"id": self._id,
                  "direction": direction,
                  "len": int(np.clip(mm, 0, 10000))
                  }
        )


def get_ground_truth_cell_coordinates():
    maze_list = [
        [
            [1253.94, -1254.48],
            [1254.0, -1087.75],
            [1254.0, -921.07],
            [1254.0, -754.4],
            [1254.0, -587.74],
            [1254.0, -421.08],
            [1253.25, -254.97],
            [1253.3, -88.23],
            [1253.41, 78.58],
            [1254.69, 246.06],
            [1254.69, 412.71],
            [1254.91, 578.23],
            [1254.96, 744.97],
            [1254.96, 911.65],
            [1254.89, 1078.41],
            [1254.93, 1245.14],
        ],
        [
            [1087.28, -1254.48],
            [1086.93, -1088.06],
            [1087.05, -921.32],
            [1087.09, -754.52],
            [1087.19, -587.75],
            [1087.26, -421.02],
            [1086.45, -254.9],
            [1086.56, -88.16],
            [1086.6, 78.64],
            [1087.88, 246.12],
            [1087.95, 412.92],
            [1088.23, 578.23],
            [1088.16, 744.97],
            [1088.23, 911.7],
            [1088.08, 1078.48],
            [1088.2, 1245.21],
        ],
        [
            [920.6, -1254.48],
            [920.2, -1087.99],
            [920.25, -921.26],
            [920.36, -754.45],
            [920.39, -587.72],
            [921.03, -420.83],
            [921.03, -254.18],
            [921.07, -87.38],
            [921.17, 79.36],
            [921.21, 246.16],
            [921.21, 412.97],
            [921.49, 578.28],
            [921.55, 745.08],
            [921.46, 911.8],
            [921.46, 1078.48],
            [921.52, 1245.21],
        ],
        [
            [753.87, -1254.44],
            [753.93, -1087.63],
            [754.04, -921.07],
            [754.1, -754.27],
            [753.74, -587.72],
            [754.22, -420.77],
            [752.83, -254.2],
            [754.34, -87.31],
            [754.37, 79.43],
            [754.48, 246.24],
            [754.53, 412.97],
            [752.47, 578.88],
            [754.87, 745.08],
            [754.66, 911.86],
            [754.72, 1078.66],
            [754.61, 1245.6],
        ],
        [
            [587.11, -1254.5],
            [587.16, -1087.77],
            [587.24, -921.0],
            [587.43, -754.27],
            [587.08, -587.72],
            [587.58, -420.77],
            [587.33, -253.58],
            [587.38, -86.84],
            [587.72, 79.43],
            [587.49, 246.62],
            [587.54, 413.35],
            [587.54, 580.03],
            [588.14, 745.13],
            [588.19, 911.93],
            [587.99, 1078.71],
            [588.03, 1245.51],
        ],
        [
            [420.44, -1254.5],
            [420.43, -1087.7],
            [420.59, -921.0],
            [420.77, -754.27],
            [420.42, -587.72],
            [420.92, -420.77],
            [420.67, -253.58],
            [420.65, -86.77],
            [420.69, 80.03],
            [420.69, 246.68],
            [420.73, 413.34],
            [420.8, 580.08],
            [418.91, 745.38],
            [421.51, 911.93],
            [421.23, 1078.82],
            [421.3, 1245.56],
        ],
        [
            [253.76, -1254.5],
            [253.88, -1087.74],
            [253.93, -921.0],
            [254.08, -754.27],
            [254.2, -587.5],
            [254.25, -420.77],
            [254.02, -253.58],
            [253.93, -86.8],
            [253.93, 79.85],
            [253.99, 246.65],
            [253.93, 413.39],
            [254.14, 580.13],
            [254.19, 746.86],
            [254.78, 911.98],
            [254.78, 1078.63],
            [254.84, 1245.43],
        ],
        [
            [87.03, -1254.43],
            [87.07, -1087.7],
            [86.86, -920.89],
            [87.35, -754.19],
            [87.39, -587.46],
            [87.43, -420.63],
            [87.47, -253.9],
            [87.47, -87.22],
            [87.47, 79.45],
            [87.31, 246.65],
            [87.28, 413.39],
            [87.34, 580.2],
            [87.46, 746.93],
            [85.42, 911.86],
            [85.2, 1079.09],
            [88.17, 1245.43],
        ],
        [
            [-79.58, -1254.6],
            [-79.51, -1087.79],
            [-79.88, -920.85],
            [-79.81, -754.05],
            [-79.43, -587.32],
            [-79.38, -420.59],
            [-79.12, -253.86],
            [-79.33, -87.23],
            [-79.26, 79.5],
            [-79.42, 246.7],
            [-79.36, 413.5],
            [-79.28, 580.2],
            [-79.23, 746.93],
            [-81.22, 911.86],
            [-81.16, 1078.66],
            [-78.49, 1245.43],
        ],
        [
            [-246.32, -1254.48],
            [-246.25, -1087.75],
            [-246.25, -921.05],
            [-246.21, -754.32],
            [-246.1, -587.32],
            [-245.85, -420.47],
            [-245.85, -253.81],
            [-245.85, -87.16],
            [-245.39, 79.68],
            [-245.34, 246.42],
            [-245.34, 413.1],
            [-246.08, 580.24],
            [-247.96, 745.3],
            [-247.96, 911.97],
            [-247.89, 1078.71],
            [-245.16, 1245.43],
        ],
        [
            [-413.12, -1254.44],
            [-413.06, -1087.64],
            [-413.06, -920.99],
            [-412.94, -754.25],
            [-412.76, -587.32],
            [-412.26, -420.34],
            [-412.19, -253.61],
            [-412.19, -86.93],
            [-412.19, 79.74],
            [-412.15, 246.42],
            [-412.08, 413.15],
            [-412.73, 580.24],
            [-414.77, 745.35],
            [-414.7, 912.16],
            [-414.7, 1078.8],
            [-411.82, 1245.43],
        ],
        [
            [-579.21, -1254.1],
            [-579.15, -1087.36],
            [-579.67, -920.98],
            [-579.62, -754.25],
            [-579.43, -587.32],
            [-578.99, -420.3],
            [-578.96, -253.5],
            [-579.0, -86.96],
            [-579.0, 79.72],
            [-578.95, 246.46],
            [-579.06, 413.23],
            [-579.39, 580.24],
            [-579.19, 746.98],
            [-579.15, 913.71],
            [-581.43, 1078.85],
            [-578.48, 1245.43],
        ],
        [
            [-745.87, -1254.1],
            [-745.89, -1087.29],
            [-746.48, -920.94],
            [-746.41, -754.14],
            [-746.09, -587.32],
            [-745.75, -420.75],
            [-745.69, -253.45],
            [-745.69, -86.8],
            [-745.65, 80.0],
            [-745.86, 246.54],
            [-745.79, 413.28],
            [-746.05, 580.24],
            [-746.0, 747.04],
            [-745.88, 913.78],
            [-748.11, 1078.85],
            [-745.13, 1245.43],
        ],
        [
            [-912.54, -1254.1],
            [-912.57, -1087.29],
            [-912.51, -920.49],
            [-912.51, -753.84],
            [-912.75, -587.32],
            [-912.39, -420.75],
            [-912.19, -254.01],
            [-912.15, -87.28],
            [-912.38, 80.05],
            [-912.38, 246.7],
            [-912.52, 413.48],
            [-912.45, 580.21],
            [-912.45, 746.89],
            [-912.56, 913.78],
            [-912.0, 1078.74],
            [-911.93, 1245.47],
        ],
        [
            [-1079.2, -1254.1],
            [-1079.43, -1087.29],
            [-1079.31, -920.53],
            [-1079.24, -753.79],
            [-1079.4, -587.32],
            [-1079.05, -420.75],
            [-1079.0, -253.95],
            [-1078.88, -87.21],
            [-1078.95, 79.35],
            [-1078.89, 246.15],
            [-1079.25, 413.52],
            [-1079.25, 580.17],
            [-1079.21, 746.98],
            [-1079.23, 913.78],
            [-1078.73, 1078.78],
            [-1078.67, 1245.59],
        ],
        [
            [-1245.87, -1254.1],
            [-1246.16, -1087.22],
            [-1246.12, -920.49],
            [-1246.2, -753.91],
            [-1246.2, -587.26],
            [-1245.63, -420.62],
            [-1245.63, -253.94],
            [-1245.57, -87.21],
            [-1245.69, 79.57],
            [-1245.69, 246.21],
            [-1245.69, 412.87],
            [-1245.69, 579.54],
            [-1245.95, 747.04],
            [-1245.89, 913.78],
            [-1245.48, 1079.0],
            [-1245.48, 1245.65],
        ],
    ]
    return np.array(maze_list)


def nearest_cell(sensors: SensorsReading, initial_guess=(0, 0), distance_threshold=5.0):
    x_sensor = sensors.down_x_offset
    y_sensor = sensors.down_y_offset
    threshold = distance_threshold

    initial_i, initial_j = initial_guess
    min_distance = float("inf")
    nearest_index = initial_guess

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()

    stack = [(initial_i, initial_j)]

    lower_i = initial_i - 2 if (initial_i - 2) >= 0 else 0
    upper_i = initial_i + 2 if (initial_i + 2) < 16 else 15

    lower_j = initial_j - 2 if (initial_j - 2) >= 0 else 0
    upper_j = initial_j + 2 if (initial_j + 2) < 16 else 15

    k = 0
    while stack:
        k += 1
        i, j = stack.pop()
        if (
            (i, j) in visited
            or not (lower_i <= i < upper_i)
            or not (lower_j <= j < upper_j)
        ):
            continue

        visited.add((i, j))

        gt = get_ground_truth_cell_coordinates()
        x_true, y_true = gt[i][j]
        distance = np.sqrt((x_sensor - x_true) ** 2 + (y_sensor - y_true) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_index = (i, j)
            if min_distance < threshold:
                return np.array(nearest_index)

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (
                (lower_i <= ni < upper_i)
                and (lower_j <= nj < upper_j)
                and (ni, nj) not in visited
            ):
                stack.append((ni, nj))

    return np.array(nearest_index)


def nearest_direction(
    sensors: SensorsReading, initial_guess: Orientation, threshold=0.01
):
    orientation_angles = {
        Orientation.NORTH: 0,
        Orientation.EAST: 90,
        Orientation.SOUTH: 180,  # or -180
        Orientation.WEST: -90,
    }

    actual_angle = sensors.rotation_yaw
    for orientation, target_angle in orientation_angles.items():
        if (
            abs(actual_angle - target_angle) <= threshold
            or abs(
                actual_angle
                - (target_angle + 360 if target_angle < 0 else target_angle - 360)
            )
            <= threshold
        ):
            return orientation

    return initial_guess


def detect_walls(
    sensors: SensorsReading, close_threshold: float
) -> Tuple[bool, bool, bool, bool]:
    front_wall = sensors.front_distance < close_threshold
    back_wall = sensors.back_distance < close_threshold
    left_wall = sensors.left_side_distance < close_threshold
    right_wall = sensors.right_side_distance < close_threshold
    return front_wall, right_wall, back_wall, left_wall


def normalize_yaw(yaw):
    normalized_yaw = yaw % 360
    if normalized_yaw > 180:
        normalized_yaw -= 360
    return normalized_yaw


def determine_cell_type(
    sensors: SensorsReading, close_threshold: float, yaw_eps: float
) -> int:
    front_wall = int(sensors.front_distance < close_threshold)
    back_wall = int(sensors.back_distance < close_threshold)
    left_wall = int(sensors.left_side_distance < close_threshold)
    right_wall = int(sensors.right_side_distance < close_threshold)

    wall_key = np.array([left_wall, front_wall, right_wall, back_wall])
    yaw = normalize_yaw(sensors.rotation_yaw)
    if abs(yaw - 0.0) < yaw_eps:
        roll_step = 0
    elif abs(yaw - 90.0) < yaw_eps:
        roll_step = 1
    elif abs(abs(yaw) - 180.0) < yaw_eps:
        roll_step = 2
    elif abs(yaw - (-90.0)) < yaw_eps:
        roll_step = -1
    else:
        roll_step = None
    wall_key = np.roll(wall_key, roll_step)

    return CELL_TYPES[tuple(wall_key)]


def update_cell(
    cell: np.ndarray,
    maze: np.ndarray,
    sensors: SensorsReading,
    close_threshold: float,
    yaw_eps: float,
    value: int = -1,
):
    if maze[cell[0], cell[1]] == value:
        maze[cell[0], cell[1]] = determine_cell_type(
            sensors, close_threshold, yaw_eps)


def cell_equals(cell1: np.ndarray, cell2: np.ndarray) -> bool:
    return (cell1 == cell2).all()


class AbstractSensorsReader(ABC):
    @abstractmethod
    def get_reading(self) -> SensorsReading:
        pass

    def start():
        pass

    def stop():
        pass


class SimpleSensorReader(AbstractSensorsReader):
    def __init__(self, api_client: SemifinalApiClient):
        super(SimpleSensorReader, self).__init__()
        self._api_client = api_client

    def get_reading(self) -> SensorsReading:
        return self._api_client.read_sensors()


def get_cell_neighbors_types(cell: np.ndarray,
                             maze: np.ndarray) -> Dict[Tuple[int, int, Orientation], int]:
    x, y = cell[0], cell[1]
    neighbors = {}
    directions = [(0, 1, Orientation.EAST),
                  (1, 0, Orientation.SOUTH),
                  (0, -1, Orientation.WEST),
                  (-1, 0, Orientation.NORTH)]
    for dx, dy, direction in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            neighbors[direction] = maze[nx, ny]
    return neighbors


def detect_cell_type_by_neighbors(cell: np.ndarray,
                                  maze: np.ndarray) -> int:
    neighbors = get_cell_neighbors_types(cell, maze)
    if -1 in neighbors.values():
        return None

    dirs = (Orientation.WEST, Orientation.NORTH,
            Orientation.EAST, Orientation.SOUTH)
    opp_dirs = (Orientation.EAST, Orientation.SOUTH,
                Orientation.WEST, Orientation.NORTH)
    sides = np.array([-1, -1, -1, -1])
    for i, (direction, opp_dir) in enumerate(zip(dirs, opp_dirs)):
        if direction in neighbors:
            sides[i] = WALLS_DICT[neighbors[direction]][opp_dir]
        else:
            sides[i] = 1
    sides = tuple(sides)
    cell_type = CELL_TYPES[sides]
    return cell_type


def refine_maze(maze: np.ndarray) -> List[np.ndarray]:
    refined_cells = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] != -1:
                continue
            else:
                cell_type = detect_cell_type_by_neighbors((i, j), maze)
                if cell_type is not None:
                    maze[i, j] = cell_type
                    refined_cells.append(np.array([i, j]))
    return refined_cells


def detect_cell_type_by_neighbors_approx(cell: np.ndarray,
                                         maze: np.ndarray) -> Optional[int]:
    neighbors = get_cell_neighbors_types(cell, maze)

    dirs = (Orientation.WEST, Orientation.NORTH,
            Orientation.EAST, Orientation.SOUTH)
    opp_dirs = (Orientation.EAST, Orientation.SOUTH,
                Orientation.WEST, Orientation.NORTH)
    sides = np.array([-1, -1, -1, -1])
    for i, (direction, opp_dir) in enumerate(zip(dirs, opp_dirs)):
        if direction in neighbors:
            if neighbors[direction] != -1:
                sides[i] = WALLS_DICT[neighbors[direction]][opp_dir]
        else:
            sides[i] = 1

    if (np.sum(sides == 1) == 3) and (np.sum(sides == -1) == 1):
        return 15
    else:
        if -1 in sides:
            return None
        else:
            sides = tuple(sides)
            cell_type = CELL_TYPES[sides]
            return cell_type


def refine_maze_approx(maze: np.ndarray) -> List[np.ndarray]:
    refined_cells = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] != -1:
                continue
            else:
                cell_type = detect_cell_type_by_neighbors_approx((i, j), maze)
                if cell_type is not None:
                    maze[i, j] = cell_type
                    refined_cells.append(np.array([i, j]))
    return refined_cells


class TremauxExplorer(AbstractMazeExplorer):
    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 walls_threshold: float,
                 yaw_eps: float,
                 heuristic: AbstractJunctionHeuristic,
                 stop_at_center: bool,
                 refine_maze: RefinementPolicy):
        super(TremauxExplorer, self).__init__()
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._walls_threshold = walls_threshold
        self._yaw_eps = yaw_eps
        self._heuristic = heuristic
        self._stop_at_center = stop_at_center
        self._refine_maze = refine_maze

        self._maze_map = np.ones((MAZE_SIDE, MAZE_SIDE), dtype=int) * -1
        self._visit_map = np.ones(
            (MAZE_SIDE, MAZE_SIDE), dtype=int) * _UNVISITED

    def run(self) -> np.ndarray:
        previous_cell = None

        while True:
            sensors_reading = self._sensors_reader.get_reading()
            current_cell = self._controller.odometry.cell
            update_cell(current_cell, self._maze_map, sensors_reading,
                        self._walls_threshold, self._yaw_eps)

            if self._stop_at_center and self._is_in_center(current_cell):
                return self._maze_map.copy(), {}

            if self._refine_maze != RefinementPolicy.NO:
                if self._refine_maze == RefinementPolicy.EXACT:
                    refined_cells = refine_maze(self._maze_map)
                else:
                    refined_cells = refine_maze_approx(self._maze_map)
                for cell in refined_cells:
                    self._visit_map[cell[0], cell[1]] = _BLOCKED

            if (self._maze_map != -1).all():
                return self._maze_map.copy(), {}

            is_dead_end, unvisited_directions, visited_directions = self._get_directions(
                sensors_reading)

            if len(unvisited_directions) == 0 and len(visited_directions) == 0:
                pass

            selected_direction = None

            if is_dead_end:
                self._visit_map[current_cell[0], current_cell[1]] = _BLOCKED
                if len(unvisited_directions) != 0:
                    selected_direction = unvisited_directions[0][0]
                else:
                    selected_direction = visited_directions[0][0]
            else:
                if len(unvisited_directions) != 0:
                    self._visit_map[current_cell[0],
                                    current_cell[1]] = _VISITED
                    selected_direction = self._heuristic.select(
                        unvisited_directions)[0]
                else:
                    backtrace_direction = None
                    for direction, direction_cell in visited_directions:
                        if (previous_cell == direction_cell).all():
                            backtrace_direction = direction
                            break
                    if backtrace_direction is not None:
                        selected_direction = backtrace_direction
                        self._visit_map[current_cell[0],
                                        current_cell[1]] = _BLOCKED
                    else:
                        selected_direction = self._heuristic.select(
                            visited_directions)[0]
                        self._visit_map[current_cell[0],
                                        current_cell[1]] += _VISITED

            previous_cell = current_cell
            self._move_robot(selected_direction)

    def _get_directions(self, sensors: SensorsReading) -> Tuple[bool,
                                                                List[Tuple[PossibleDirection,
                                                                           np.ndarray]],
                                                                List[Tuple[PossibleDirection,
                                                                           np.ndarray]],
                                                                int]:
        front_wall, right_wall, back_wall, left_wall = detect_walls(
            sensors, self._walls_threshold)
        can_front = not front_wall
        can_right = not right_wall
        can_back = not back_wall
        can_left = not left_wall

        is_dead_end = sum([front_wall, right_wall, back_wall, left_wall]) > 2

        odom = self._controller.odometry
        visited_directions = []
        unvisited_directions = []

        if can_front:
            next_cell = odom.move_forward().cell
            visit_value = self._check_next_cell(next_cell)
            direction = (PossibleDirection.FRONT, next_cell)
            if visit_value == _UNVISITED:
                unvisited_directions.append(direction)
            elif visit_value == _VISITED:
                visited_directions.append(direction)

        if can_right:
            next_cell = odom.rotatate_right().move_forward().cell
            visit_value = self._check_next_cell(next_cell)
            direction = (PossibleDirection.RIGHT, next_cell)
            if visit_value == _UNVISITED:
                unvisited_directions.append(direction)
            elif visit_value == _VISITED:
                visited_directions.append(direction)

        if can_left:
            next_cell = odom.rotate_left().move_forward().cell
            visit_value = self._check_next_cell(next_cell)
            direction = (PossibleDirection.LEFT, next_cell)
            if visit_value == _UNVISITED:
                unvisited_directions.append(direction)
            elif visit_value == _VISITED:
                visited_directions.append(direction)

        if can_back:
            next_cell = odom.move_backward().cell
            visit_value = self._check_next_cell(next_cell)
            direction = (PossibleDirection.BACK, next_cell)
            if visit_value == _UNVISITED:
                unvisited_directions.append(direction)
            elif visit_value == _VISITED:
                visited_directions.append(direction)

        return is_dead_end, unvisited_directions, visited_directions

    def _check_next_cell(self, next_cell: np.ndarray) -> bool:
        if (next_cell >= MAZE_SIDE).any() or (next_cell < 0).any():
            return False
        visit_value = self._visit_map[next_cell[0], next_cell[1]]
        return visit_value

    def _move_robot(self, direction: PossibleDirection):
        if direction == PossibleDirection.FRONT:
            self._controller.move_forward()
        elif direction == PossibleDirection.BACK:
            self._controller.move_backward()
        elif direction == PossibleDirection.RIGHT:
            self._controller.rotate_right()
            self._controller.move_forward()
        elif direction == PossibleDirection.LEFT:
            self._controller.rotate_left()
            self._controller.move_forward()
        else:
            pass

    def _is_in_center(self, cell: np.ndarray) -> bool:
        cell = (int(cell[0]), int(cell[1]))
        return cell in ((7, 7),
                        (7, 8),
                        (8, 7),
                        (8, 8))


class BFSPlanner(AbstractPlanner):

    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 maze: np.ndarray):
        super(BFSPlanner, self).__init__(controller, sensors_reader, maze)

    def find_path(self, start_cell: np.ndarray, target_cell: np.ndarray) -> np.ndarray:
        queue = deque([[(int(start_cell[0]), int(start_cell[1]))]])
        visited = set([(int(start_cell[0]), int(start_cell[1]))])
        target_cell = (int(target_cell[0]), int(target_cell[1]))

        path = None
        while queue:
            path = queue.popleft()
            cell = path[-1]
            if cell == target_cell:
                break
            for neighbor in self._get_neighbors(cell):
                neighbor = (int(neighbor[0]), int(neighbor[1]))
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        path = np.array(path)
        return path

    def _get_neighbors(self, cell: np.ndarray):
        x, y = cell[0], cell[1]
        neighbors = []
        directions = [(0, 1, Orientation.EAST),
                      (1, 0, Orientation.SOUTH),
                      (0, -1, Orientation.WEST),
                      (-1, 0, Orientation.NORTH)]
        cell_value = self._maze[x][y]

        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._maze.shape[0] and 0 <= ny < self._maze.shape[1]:
                if cell_value != -1 and not self._has_wall(cell_value, direction):
                    neighbors.append((nx, ny))

        return neighbors

    def _has_wall(self, cell_value: int, direction: Orientation):
        return WALLS_DICT[cell_value][direction]


def main(api_token: str = API_TOKEN):
    client = SemifinalApiClient(robot_ip=ROBOT_IP,
                                id=ROBOT_ID,
                                max_retries=10,
                                timeout=5.,
                                sleep_time=0.05)
    sensors_reader = SimpleSensorReader(api_client=client)

    controller = SemifinalBasicRobotController(api_client=client,
                                               delay=0.010)
    explorer = TremauxExplorer(controller=controller,
                               sensors_reader=sensors_reader,
                               walls_threshold=101.0,
                               yaw_eps=1.,
                               heuristic=ManhattanHeuristic(
                                   target_point=np.array([7.5, 7.5])
                               ),
                               stop_at_center=True,
                               refine_maze=RefinementPolicy.APPROX)
    maze, _ = explorer.run()


if __name__ == "__main__":
    main()
