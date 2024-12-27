import numpy as np

from typing import Tuple, Dict, List, Optional
from lib.maze.common import CELL_TYPES
from lib.entities.sensors import SensorsReading
from lib.entities.state import Orientation
from lib.maze.common import WALLS_DICT


def get_ground_truth_cell_coordinates():
    maze_list = [[[1253.94, -1254.48],
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
                  [1254.93, 1245.14]],
                 [[1087.28, -1254.48],
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
                  [1088.2, 1245.21]],
                 [[920.6, -1254.48],
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
                  [921.52, 1245.21]],
                 [[753.87, -1254.44],
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
                  [754.61, 1245.6]],
                 [[587.11, -1254.5],
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
                  [588.03, 1245.51]],
                 [[420.44, -1254.5],
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
                  [421.3, 1245.56]],
                 [[253.76, -1254.5],
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
                  [254.84, 1245.43]],
                 [[87.03, -1254.43],
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
                  [88.17, 1245.43]],
                 [[-79.58, -1254.6],
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
                  [-78.49, 1245.43]],
                 [[-246.32, -1254.48],
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
                  [-245.16, 1245.43]],
                 [[-413.12, -1254.44],
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
                  [-411.82, 1245.43]],
                 [[-579.21, -1254.1],
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
                  [-578.48, 1245.43]],
                 [[-745.87, -1254.1],
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
                  [-745.13, 1245.43]],
                 [[-912.54, -1254.1],
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
                  [-911.93, 1245.47]],
                 [[-1079.2, -1254.1],
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
                  [-1078.67, 1245.59]],
                 [[-1245.87, -1254.1],
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
                  [-1245.48, 1245.65]]]
    return np.array(maze_list)


def nearest_cell(sensors: SensorsReading, initial_guess=(0, 0), distance_threshold=5.0):
    x_sensor = sensors.down_x_offset
    y_sensor = sensors.down_y_offset
    threshold = distance_threshold

    initial_i, initial_j = initial_guess
    min_distance = float('inf')
    nearest_index = initial_guess

    # Directions for potential neighbors (4-way: up, down, left, right)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = set()

    # Stack for Depth-First Search (DFS) approach starting from the initial guess
    stack = [(initial_i, initial_j)]

    # Look for the nearest cell in 5x5 rectangle without border clipping
    lower_i = (initial_i - 2 if (initial_i - 2) >= 0 else 0)
    upper_i = (initial_i + 2 if (initial_i + 2) < 16 else 15)

    lower_j = (initial_j - 2 if (initial_j - 2) >= 0 else 0)
    upper_j = (initial_j + 2 if (initial_j + 2) < 16 else 15)

    k = 0
    while stack:
        k += 1
        i, j = stack.pop()
        if (i, j) in visited or not (lower_i <= i < upper_i) or not (lower_j <= j < upper_j):
            continue

        visited.add((i, j))

        gt = get_ground_truth_cell_coordinates()
        x_true, y_true = gt[i][j]
        # Calculate the Euclidean distance
        distance = np.sqrt((x_sensor - x_true)**2 + (y_sensor - y_true)**2)

        if distance < min_distance:
            min_distance = distance
            nearest_index = (i, j)
            if (min_distance < threshold):
                # Return early - we are most probably already found the cell
                return np.array(nearest_index)

        # Add neighbors to the stack
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (lower_i <= ni < upper_i) and (lower_j <= nj < upper_j) and (ni, nj) not in visited:
                stack.append((ni, nj))

    return np.array(nearest_index)


def nearest_direction(sensors: SensorsReading, initial_guess: Orientation, threshold=0.01):
    # Define the angles corresponding to each orientation in degrees
    orientation_angles = {
        Orientation.NORTH: 0,
        Orientation.EAST: 90,
        Orientation.SOUTH: 180,  # or -180
        Orientation.WEST: -90
    }

    actual_angle = sensors.rotation_yaw
    for orientation, target_angle in orientation_angles.items():
        # Check the angle against the target angle and its equivalent positive representation, if necessary
        if abs(actual_angle - target_angle) <= threshold or abs(actual_angle - (target_angle + 360 if target_angle < 0 else target_angle - 360)) <= threshold:
            return orientation

    return initial_guess


def detect_walls(sensors: SensorsReading, close_threshold: float) -> Tuple[bool, bool, bool, bool]:
    """Detects if there are walls along the orthogonal directions to the robot

    Args:
        sensors (SensorsReading): Sensors reading to use for detection
        close_threshold (float): The threshold distance, if 
                                 the measurement is lower than it, the wall is detected

    Returns:
        Tuple[bool, bool, bool, bool]: Presence of walls in order: front, right, back, left (relative robot frame)
    """
    front_wall = sensors.front_distance < close_threshold
    back_wall = sensors.back_distance < close_threshold
    left_wall = sensors.left_side_distance < close_threshold
    right_wall = sensors.right_side_distance < close_threshold
    return front_wall, right_wall, back_wall, left_wall


def determine_cell_type(sensors: SensorsReading,
                        close_threshold: float, 
                        yaw_eps: float,
                        semifinal_mode: bool) -> int:
    """Determines cell type based on the sensors measurements.

    Args:
        sensors (SensorsReading): Sensors reading to use for wall detection
        close_threshold (float): The threshold distance, if 
                                 the measurement is lower than it, the wall is detected
        yaw_eps (float): Angular difference tolerance to detect current orientation

    Returns:
        int: Index of the cell type
    """
    front_wall = int(sensors.front_distance < close_threshold)
    back_wall = int(sensors.back_distance < close_threshold)
    left_wall = int(sensors.left_side_distance < close_threshold)
    right_wall = int(sensors.right_side_distance < close_threshold)

    wall_key = np.array([left_wall, front_wall, right_wall, back_wall])
    yaw = sensors.rotation_yaw
    
    if semifinal_mode:
        if abs(yaw - 0.) < yaw_eps:
            roll_step = 0
        elif abs(yaw - 90.) < yaw_eps:
            roll_step = 1
        elif abs(abs(yaw) - 180.) < yaw_eps:
            roll_step = 2
        elif abs(yaw - 270.) < yaw_eps:
            roll_step = -1
        else:
            roll_step = None
    else:
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


def update_cell(cell: np.ndarray,
                maze: np.ndarray,
                sensors: SensorsReading,
                close_threshold: float,
                yaw_eps: float,
                semifinal_mode: bool,
                value: int = -1):
    """Updates the cell type in the maze in-place.

    Args:
        cell (np.ndarray): Cell coordinates to update
        maze (np.ndarray): Maze to update
        sensors (SensorsReading): Sensors reading to use for wall detection
        close_threshold (float): The threshold distance, if 
                                 the measurement is lower than it, the wall is detected
        yaw_eps (float): Angular difference tolerance to detect current orientation
        value (int, optional): Value to compare with. Defaults to -1. 
                               For Flood Fill it is required to set to 0 to initialize the maze with 0 index walls.
    """
    if maze[cell[0], cell[1]] == value:
        maze[cell[0], cell[1]] = determine_cell_type(sensors, close_threshold, yaw_eps, semifinal_mode)


def cell_equals(cell1: np.ndarray,
                cell2: np.ndarray) -> bool:
    return (cell1 == cell2).all()


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
        # else:
        #     neighbors[direction] = None
    return neighbors


def detect_cell_type_by_neighbors(cell: np.ndarray, 
                                  maze: np.ndarray) -> int:
    neighbors = get_cell_neighbors_types(cell, maze)
    if -1 in neighbors.values():
        return None
    
    dirs = (Orientation.WEST, Orientation.NORTH, Orientation.EAST, Orientation.SOUTH)
    opp_dirs = (Orientation.EAST, Orientation.SOUTH, Orientation.WEST, Orientation.NORTH)
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
        
    dirs = (Orientation.WEST, Orientation.NORTH, Orientation.EAST, Orientation.SOUTH)
    opp_dirs = (Orientation.EAST, Orientation.SOUTH, Orientation.WEST, Orientation.NORTH)
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
