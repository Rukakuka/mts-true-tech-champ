import enum
import requests
import numpy as np
import time

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


API_TOKEN = "44f20028-0138-410c-a1cd-c660dc8d15c466cb1331-67e8-4897-a192-c5ef202068e2"


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
    FORWARD = 0
    RIGHT = 1
    BACKWARD = 2
    LEFT = 3


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


class RoboController:

    _BASE_URL = "http://localhost:8801/api/v1/robot-cells"
    _MATRIX_URL = "http://localhost:8801/api/v1/matrix/send"

    def __init__(self, api_token: str, delay: float = 0.1) -> None:
        self._token = api_token
        self._current_cell = (15, 0)
        self._current_orientation = Orientation.FORWARD
        reading = self.read_sensors()
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        self._delay = delay
    
    @property
    def current_cell(self) -> Tuple[int, int]:
        return self._current_cell
    
    @property
    def current_orientation(self) -> Orientation:
        return self._current_orientation

    def forward(self) -> bool:
        reading = self.read_sensors()
        # if abs(reading.down_x_offset - self._last_offset[0]) < 10. and abs(reading.down_y_offset - self._last_offset[1]) < 10.:
        #     return False
        if reading.front_distance < 65.:
            return False
        self._movement_request("forward")
        if self._current_orientation == Orientation.FORWARD:
            d1, d2 = -1, 0
        elif self._current_orientation == Orientation.BACKWARD:
            d1, d2 = 1, 0
        elif self._current_orientation == Orientation.RIGHT:
            d1, d2 = 0, 1
        else:
            d1, d2 = 0, -1
        self._current_cell = (self._current_cell[0] + d1, self._current_cell[1] + d2)
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        return True
    
    def backward(self) -> bool:
        reading = self.read_sensors()
        # if abs(reading.down_x_offset - self._last_offset[0]) < 10. and abs(reading.down_y_offset - self._last_offset[1]) < 10.:
        #     return False
        if reading.back_distance < 65.:
            return False
        self._movement_request("backward")
        if self._current_orientation == Orientation.FORWARD:
            d1, d2 = 1, 0
        elif self._current_orientation == Orientation.BACKWARD:
            d1, d2 = -1, 0
        elif self._current_orientation == Orientation.RIGHT:
            d1, d2 = 0, -1
        else:
            d1, d2 = 0, 1
        self._current_cell = (self._current_cell[0] + d1, self._current_cell[1] + d2)
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        return True
    
    def right(self) -> bool:
        self._movement_request("right")
        reading = self.read_sensors()
        if self._current_orientation == Orientation.FORWARD:
            self._current_orientation = Orientation.RIGHT
        elif self._current_orientation == Orientation.BACKWARD:
            self._current_orientation = Orientation.LEFT
        elif self._current_orientation == Orientation.RIGHT:
            self._current_orientation = Orientation.BACKWARD
        else:
            self._current_orientation = Orientation.FORWARD
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        return True
    
    def left(self) -> bool:
        self._movement_request("left")
        reading = self.read_sensors()
        if self._current_orientation == Orientation.FORWARD:
            self._current_orientation = Orientation.LEFT
        elif self._current_orientation == Orientation.BACKWARD:
            self._current_orientation = Orientation.RIGHT
        elif self._current_orientation == Orientation.RIGHT:
            self._current_orientation = Orientation.FORWARD
        else:
            self._current_orientation = Orientation.BACKWARD
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        return True
    
    def read_sensors(self, max_retries = 10, timeout = 5) -> SensorsReading:
        url = f"{RoboController._BASE_URL}/sensor-data?token={self._token}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                break 
            except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, requests.exceptions.RequestException) as err:
                print(f"Attempt {attempt + 1} failed: {err}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(0.05)
                else:
                    print("Max retries reached. Exiting.")

        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        return SensorsReading(**response.json())


    def _movement_request(self, direction: str,  max_retries:int = 10, timeout:int = 5) -> bool:
        url = f"{RoboController._BASE_URL}/{direction}?token={self._token}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, timeout=timeout)
                response.raise_for_status()
                break 
            except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, requests.exceptions.RequestException) as err:
                print(f"Attempt {attempt + 1} failed: {err}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(0.05)
                else:
                    print("Max retries reached. Exiting.")

        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        time.sleep(self._delay)
        return True

    def send_matrix(self, matrix) -> bool:
        url = f"{RoboController._MATRIX_URL}?token={self._token}"
        try:
            response = requests.post(url, json=matrix)
            if response.status_code != 200:
                raise RuntimeError(f"Received status code {response.status_code}")
            return True
        finally:
            time.sleep(self._delay)
    

def determine_walls(sensors: SensorsReading, close_threshold: float = 65.) -> Tuple[bool, bool, bool, bool]:
    front_wall = sensors.front_distance < close_threshold
    back_wall = sensors.back_distance < close_threshold
    left_wall = sensors.left_side_distance < close_threshold
    right_wall = sensors.right_side_distance < close_threshold
    return left_wall, front_wall, right_wall, back_wall


def determine_cell_type(sensors: SensorsReading, close_threshold: float = 65., yaw_eps: float = 1.) -> int:
    front_wall = int(sensors.front_distance < close_threshold)
    back_wall = int(sensors.back_distance < close_threshold)
    left_wall = int(sensors.left_side_distance < close_threshold)
    right_wall = int(sensors.right_side_distance < close_threshold)

    wall_key = np.array([left_wall, front_wall, right_wall, back_wall])
    yaw = sensors.rotation_yaw
    if abs(yaw - 0.) < yaw_eps:
        roll_step = 0
    elif abs(yaw - 90.) < yaw_eps:
        roll_step = 1
    elif abs(abs(yaw) - 180.) < yaw_eps:
        roll_step = 2
    elif abs(yaw - (-90.)) < yaw_eps:
        roll_step = -1
    else:
        roll_step = None
    wall_key = np.roll(wall_key, roll_step)

    return CELL_TYPES[tuple(wall_key)]


controller = RoboController(API_TOKEN)


def update_cell(controller: RoboController,
                maze: np.ndarray):
    current_cell = controller.current_cell
    if maze[current_cell[0], current_cell[1]] == -1:
        maze[current_cell[0], current_cell[1]] = determine_cell_type(controller.read_sensors())


def explore_step(controller: RoboController,
                 maze: np.ndarray,
                 backtrack: Optional[List[str]] = None) -> bool:
    if not (maze == -1).any():
        return

    current_cell = controller.current_cell
    # print(current_cell)
    sensors = controller.read_sensors()
    if maze[current_cell[0], current_cell[1]] == -1:
        maze[current_cell[0], current_cell[1]] = determine_cell_type(controller.read_sensors())

    left, front, right, back = determine_walls(sensors)
    # Convert walls presence to direction possibility
    left = not left
    front = not front
    right = not right
    back = not back

    # We don't want to go back to the explored cells
    if right:
        if controller.current_orientation == Orientation.FORWARD:
            next_cell = (controller._current_cell[0], controller.current_cell[1] + 1)
        elif controller.current_orientation == Orientation.RIGHT:
            next_cell = (controller._current_cell[0] + 1, controller.current_cell[1])
        elif controller.current_orientation == Orientation.BACKWARD:
            next_cell = (controller._current_cell[0], controller.current_cell[1] - 1)
        else:
            next_cell = (controller._current_cell[0] - 1, controller.current_cell[1])
        if next_cell[0] < 16 and next_cell[0] > 0 and next_cell[1] < 16 and next_cell[0] > 0:
            if maze[next_cell[0], next_cell[1]] != -1:
                right = False

    if left:
        if controller.current_orientation == Orientation.FORWARD:
            next_cell = (controller._current_cell[0], controller.current_cell[1] - 1)
        elif controller.current_orientation == Orientation.RIGHT:
            next_cell = (controller._current_cell[0] - 1, controller.current_cell[1])
        elif controller.current_orientation == Orientation.BACKWARD:
            next_cell = (controller._current_cell[0], controller.current_cell[1] + 1)
        else:
            next_cell = (controller._current_cell[0] + 1, controller.current_cell[1])
        if next_cell[0] < 16 and next_cell[0] > 0 and next_cell[1] < 16 and next_cell[0] > 0:
            if maze[next_cell[0], next_cell[1]] != -1:
                left = False

    # Case 1: We can move only forward
    if front and (not left) and (not right):
        controller.forward()
        if backtrack is not None:
            backtrack.append("forward")
        return explore_step(controller, maze, backtrack)
    
    # Case 2: The only turn is to the right
    elif right and (not front) and (not left):
        controller.right()
        controller.forward()
        if backtrack is not None:
            backtrack.append("right")
            backtrack.append("forward")
        return explore_step(controller, maze, backtrack)
    
    # Case 3: The only turn is to the left
    elif left and (not front) and (not right):
        controller.left()
        controller.forward()
        if backtrack is not None:
            backtrack.append("left")
            backtrack.append("forward")
        return explore_step(controller, maze, backtrack)
    
    # Case 4: We stuck, just unravel the backtrack
    if (not left) and (not front) and (not right):
        if backtrack is None or len(backtrack) == 0:
            raise ValueError("Robot stuck and backtrack is empty!")
        for action in backtrack[::-1]:
            if action == "forward":
                controller.backward()
            elif action == "backward":
                controller.forward()
            elif action == "left":
                controller.right()
            else:
                controller.left()
        return

    # Case 5: Multiple ways to explore
    if front:
        controller.forward()
        sub_backtrack = ["forward"]
        explore_step(controller, maze, sub_backtrack)
    if right:
        controller.right()
        controller.forward()
        sub_backtrack = ["right", "forward"]
        explore_step(controller, maze, sub_backtrack)
    if left:
        controller.left()
        controller.forward()
        sub_backtrack = ["left", "forward"]
        explore_step(controller, maze, sub_backtrack)

    if not (maze == -1).any():
        return
    
    if backtrack is not None:
        for action in backtrack[::-1]:
            if action == "forward":
                controller.backward()
            elif action == "backward":
                controller.forward()
            elif action == "left":
                controller.right()
            else:
                controller.left()
    return

maze = np.ones((16, 16), dtype=int) * -1


explore_step(controller, maze, None)

print((maze == -1).any())

print(maze.tolist())

controller.send_matrix(maze.tolist())

