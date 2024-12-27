import enum
import requests
import numpy as np
import time

from typing import Dict, Tuple, List, Optional
from collections import deque

from dataclasses import dataclass


API_TOKEN = "288d4eac-0223-4547-a03e-d5acbf6264639ba2339b-0908-43a6-8379-f2f199a6b036"


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
WALLS_DICT = {
        0: {
            'W': False,
            'N': False,
            'E': False,
            'S': False 
        },
        1: {
            'W': True,
            'N': False,
            'E': False,
            'S': False 
        },
        2: {
            'W': False,
            'N': True,
            'E': False,
            'S': False 
        },
        3: {
            'W': False,
            'N': False,
            'E': True,
            'S': False 
        },
        4: {
            'W': False,
            'N': False,
            'E': False,
            'S': True 
        },
        5: {
            'W': True,
            'N': False,
            'E': False,
            'S': True 
        },
        6: {
            'W': False,
            'N': False,
            'E': True,
            'S': True 
        },
        7: {
            'W': False,
            'N': True,
            'E': True,
            'S': False 
        },
        8: {
            'W': True,
            'N': True,
            'E': False,
            'S': False 
        },
        9: {
            'W': True,
            'N': False,
            'E': True,
            'S': False 
        },
        10: {
            'W': False,
            'N': True,
            'E': False,
            'S': True 
        },
        11: {
            'W': False,
            'N': True,
            'E': True,
            'S': True 
        },
        12: {
            'W': True,
            'N': True,
            'E': True,
            'S': False 
        },
        13: {
            'W': True,
            'N': True,
            'E': False,
            'S': True 
        },
        14: {
            'W': True,
            'N': False,
            'E': True,
            'S': True 
        },
        15: {
            'W': True,
            'N': True,
            'E': True,
            'S': True 
        }
    }


class Orientation(enum.Enum):
    FORWARD = "N"
    RIGHT = "E"
    BACKWARD = "S"
    LEFT = "W"


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
    
    def read_sensors(self) -> SensorsReading:
        url = f"{RoboController._BASE_URL}/sensor-data?token={self._token}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        return SensorsReading(**response.json())

    def reset_maze(self):
        url = f"http://127.0.0.1:8801/api/v1/maze/restart?token={self._token}"
        response = requests.post(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        time.sleep(self._delay)
        return True

    def _movement_request(self, direction: str) -> bool:
        url = f"{RoboController._BASE_URL}/{direction}?token={self._token}"
        response = requests.post(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        time.sleep(self._delay)
        return True
    

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


def update_cell(controller: RoboController,
                maze: np.ndarray):
    current_cell = controller.current_cell
    if maze[current_cell[0], current_cell[1]] == -1:
        maze[current_cell[0], current_cell[1]] = determine_cell_type(controller.read_sensors())


def explore_step(controller: RoboController,
                 maze: np.ndarray,
                 heuristic_enabled: bool,
                 backtrack: Optional[List[str]] = None,
                 eps_prob: float = 0.1) -> bool:
    
    centrall_cells = [(7,7), (7,8), (8,7), (8,8)]
    if not (maze == -1).any():
        return True

    current_cell = controller.current_cell
    print(f"Current state: {current_cell}")
    # if current_cell == (7, 7):
    #     return True

    sensors = controller.read_sensors()
    if maze[current_cell[0], current_cell[1]] == -1:
        maze[current_cell[0], current_cell[1]] = determine_cell_type(controller.read_sensors())

    left, front, right, back = determine_walls(sensors)
    # Convert walls presence to direction possibility
    left = not left
    front = not front
    right = not right
    back = not back

    next_cell_right = (100., 100.)
    next_cell_left = (100., 100.)
    next_cell_front = (100., 100.)


    # We don't want to go back to the explored cells
    if front:
        if controller.current_orientation == Orientation.FORWARD:
            next_cell = (controller._current_cell[0] - 1, controller.current_cell[1])
        elif controller.current_orientation == Orientation.RIGHT:
            next_cell = (controller._current_cell[0], controller.current_cell[1] + 1)
        elif controller.current_orientation == Orientation.BACKWARD:
            next_cell = (controller._current_cell[0] + 1, controller.current_cell[1])
        else:
            next_cell = (controller._current_cell[0], controller.current_cell[1] - 1)
        # if next_cell[0] < 16 and next_cell[0] > 0 and next_cell[1] < 16 and next_cell[0] > 0:
        if 0 <= next_cell[0] < 16 and 0 <= next_cell[1] < 16:
            if next_cell in centrall_cells or maze[next_cell[0], next_cell[1]] != -1:
                front = False
            else:
                next_cell_front = next_cell

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
            if next_cell in centrall_cells or maze[next_cell[0], next_cell[1]] != -1:
                right = False
            else:
                next_cell_right = next_cell

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
            if next_cell in centrall_cells or maze[next_cell[0], next_cell[1]] != -1:
                left = False
            else:
                next_cell_left = next_cell

    # Case 1: We can move only forward
    if front and (not left) and (not right):
        controller.forward()
        if backtrack is not None:
            backtrack.append("forward")
        return explore_step(controller, maze, heuristic_enabled, backtrack, eps_prob)
    
    # Case 2: The only turn is to the right
    elif right and (not front) and (not left):
        controller.right()
        controller.forward()
        if backtrack is not None:
            backtrack.append("right")
            backtrack.append("forward")
        return explore_step(controller, maze, heuristic_enabled, backtrack, eps_prob)
    
    # Case 3: The only turn is to the left
    elif left and (not front) and (not right):
        controller.left()
        controller.forward()
        if backtrack is not None:
            backtrack.append("left")
            backtrack.append("forward")
        return explore_step(controller, maze, heuristic_enabled, backtrack, eps_prob)
    
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
        return False

    # Case 5: Multiple ways to explore
    directions = []
    cells = []
    if front:
        directions.append("front")
        cells.append(next_cell_front)
    if right:
        directions.append("right")
        cells.append(next_cell_right)
    if left:
        directions.append("left")
        cells.append(next_cell_left)
    directions = np.array(directions)
    cells = np.array(cells)

    resampled = False
    if eps_prob is not None:
        if bool(np.random.binomial(1, eps_prob)):
            np.random.shuffle(directions)
            resampled = True
    
    if heuristic_enabled and (not resampled):
        distances = np.linalg.norm(cells - np.array([7., 7.]), ord=1, axis=1)
        indicies = np.argsort(distances)
        directions = directions[indicies]

    for direction in directions:
        if direction == "front":
            controller.forward()
            sub_backtrack = ["forward"]
            result = explore_step(controller, maze, heuristic_enabled, sub_backtrack, eps_prob)
            if result:
                return True
        elif direction == "right":
            controller.right()
            controller.forward()
            sub_backtrack = ["right", "forward"]
            result = explore_step(controller, maze, heuristic_enabled, sub_backtrack, eps_prob)
            if result:
                return True
        elif direction == "left":
            controller.left()
            controller.forward()
            sub_backtrack = ["left", "forward"]
            result = explore_step(controller, maze, heuristic_enabled, sub_backtrack, eps_prob)
            if result:
                return True
        else:
            continue

    if not (maze == -1).any():
        return True
    if controller.current_cell == (7, 7):
        return True
    
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
    return False

def find_enter(data: np.ndarray) -> Tuple[int,int]:

    centrall_cells = [(7,7), (7,8), (8,7),  (8,8)]
    
    dict_enter = {

        (7,6): [0,1,2,4,5,8,10,13],
        (8,6): [0,1,2,4,5,8,10,13],

        (6,7): [0,1,2,3,7,8,9,12],
        (6,8): [0,1,2,3,7,8,9,12],

        (9,7): [0,1,3,4,5,6,9,14],
        (9,8): [0,1,3,4,5,6,9,14],

        (7,9): [0,2,3,4,6,7,10,11],
        (8,9): [0,2,3,4,6,7,10,11],
    }


    def is_entry(cell_init) -> bool:
        x,y = cell_init
        neighbours_list = [(x-1,y), (x+1,y), (x,y-1), (x, y+1)]
        #print(f"neighbours_list for {cell_init}: {neighbours_list}")
        neighbour_list_types = []
        for cell in neighbours_list:
            x, y = cell
            neighbour_list_types.append(data[x,y])
        #print(f"neighbours_list_types for {cell}: {neighbour_list_types}")

        neighbours_list = np.array(neighbours_list)
        neighbour_list_types = np.array(neighbour_list_types)

        neighbours_list_filtered = neighbours_list[neighbour_list_types != -1]
        neighbours_list_filtered_tuples = [tuple(x) for x in neighbours_list_filtered]
        #print(f"neighbours_list_filtered for {cell_init}: {neighbours_list_filtered_tuples}")
        for neighbour in neighbours_list_filtered_tuples:
            x,y = neighbour
            value = data[x,y,]
            #print(f"x = {x},y = {y}, val = {value}")
            if neighbour in dict_enter and value in dict_enter[neighbour]:
                return True

    for central_cell in centrall_cells:
        if is_entry(central_cell):
            return central_cell
        
def get_neighbors(cell, maze):
    x, y = cell
    neighbors = []
    directions = [(0, 1, 'E'), (1, 0, 'S'), (0, -1, 'W'), (-1, 0, 'N')]
    cell_value = maze[x][y]
    
    for dx, dy, direction in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            if cell_value != -1 and not has_wall(cell_value, direction):
                neighbors.append((nx, ny))
    
    return neighbors

def has_wall(cell_value, direction):
    return WALLS_DICT[cell_value][direction]

def bfs_shortest_path(start, goal, maze):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        cell = path[-1]
        
        if cell == tuple(goal):
            return path
        
        for neighbor in get_neighbors(cell, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None

def navigate_to_center(path,controller: RoboController):
    global current_position, current_orientation
    
    current_position = controller.current_cell

    current_orientation = controller.current_orientation.value

    orientations = ['N', 'E', 'S', 'W']
    
    for i in range(1, len(path)):
        prev_cell, curr_cell = path[i-1], path[i]
        dx, dy = curr_cell[0] - prev_cell[0], curr_cell[1] - prev_cell[1]
        
        if dx == 1:
            target_orientation = 'S'
        elif dx == -1:
            target_orientation = 'N'
        elif dy == 1:
            target_orientation = 'E'
        else:
            target_orientation = 'W'
        
        # Determine the most efficient way to turn
        current_index = orientations.index(current_orientation)
        target_index = orientations.index(target_orientation)
        
        turn_difference = (target_index - current_index + 4) % 4
        
        if turn_difference == 0:
            # Already facing the right direction, no need to turn
            pass
        elif turn_difference == 1:
            
            controller.right()
            # turn_right()
        elif turn_difference == 2:
            # It's faster to move backward
            pass
        else:  # turn_difference == 3
            #turn_left()
            controller.left()

        current_orientation = target_orientation
        
        # Move in the correct direction
        if turn_difference == 2:
            controller.backward()
        else:
            controller.forward()
        
        current_position = list(curr_cell)
        time.sleep(0.1)
    
    print(f"Reached the center at {current_position}")



# data = np.load(file="maze.npy")
# test = find_enter(data)

def main():
    controller = RoboController(API_TOKEN)

    maze = np.ones((16, 16), dtype=int) * -1

    explore_step(controller, maze, True, None, 0.1)

    print((maze == -1).any())
    print(maze)

    controller.reset_maze()
    # maze = np.load("maze.npy")

    goal_point = find_enter(data=maze)
    
    path = bfs_shortest_path((15, 0), goal_point, maze)
    navigate_to_center(path=path,controller=controller)

    # np.save("maze.npy", maze)


if __name__ == "__main__":
    main()
