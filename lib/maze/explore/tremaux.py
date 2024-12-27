import numpy as np

from typing import List, Tuple
from lib.maze.explore.algo_base import AbstractMazeExplorer, RefinementPolicy
from lib.entities.sensors import SensorsReading
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import AbstractSensorsReader
from lib.maze.common import MAZE_SIDE, PossibleDirection
from lib.maze.util import detect_walls, update_cell, refine_maze, refine_maze_approx
from lib.maze.explore.heuristic import AbstractJunctionHeuristic


_UNVISITED = 0
_VISITED = 1
_BLOCKED = 2 * _VISITED


class TremauxExplorer(AbstractMazeExplorer):
    """Tremaux algorithm implementation for the unknown maze exploration.
    See references:
    https://en.wikipedia.org/wiki/Maze-solving_algorithm
    https://github.com/mnmnc/MachineLearning_Maze/blob/master/tremaux.py
    https://github.com/illiterati1/python_maze/blob/master/tremaux.py
    """

    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 walls_threshold: float,
                 yaw_eps: float,
                 heuristic: AbstractJunctionHeuristic,
                 stop_at_center: bool,
                 refine_maze: RefinementPolicy,
                 semifinal_mode: bool):
        """Instanitates the explorer.

        Args:
            controller (AbstractRobotController): Robot controller
            sensors_reader (AbstractSensorsReader): Sensors reader
            walls_threshold (float): Minimal distance to the wall at 
                                     which wall is considered as detected 
                                     and movement is forbidden in corresponding direction
            yaw_eps (float): Tolerance of the yaw orientation quadrant
            heuristic (AbstractJunctionHeuristic): Heuristic for direction selection at the junctions
            stop_at_center (bool): If true, exploration is stopped when robot entrance the central region
        """
        super(TremauxExplorer, self).__init__()
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._walls_threshold = walls_threshold
        self._yaw_eps = yaw_eps
        self._heuristic = heuristic
        self._stop_at_center = stop_at_center
        self._refine_maze = refine_maze
        self._semifinal_mode = semifinal_mode
        
        self._maze_map = np.ones((MAZE_SIDE, MAZE_SIDE), dtype=int) * -1
        self._visit_map = np.ones((MAZE_SIDE, MAZE_SIDE), dtype=int) * _UNVISITED


    def run(self) -> np.ndarray:
        """Run the exploration (mapping) process.

        Returns:
            np.ndarray: built map of the maze
        """
        previous_cell = None

        while True:
            sensors_reading = self._sensors_reader.get_reading()
            current_cell = self._controller.odometry.cell
            update_cell(current_cell, self._maze_map, sensors_reading,
                        self._walls_threshold, self._yaw_eps, self._semifinal_mode)
            
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
            
            is_dead_end, unvisited_directions, visited_directions = self._get_directions(sensors_reading)

            if len(unvisited_directions) == 0 and len(visited_directions) == 0:
                raise RuntimeError("No directions!")
            
            selected_direction = None

            # Case 1: true dead end
            if is_dead_end:
                self._visit_map[current_cell[0], current_cell[1]] = _BLOCKED
                if len(unvisited_directions) != 0:
                    # Special case for beginning
                    selected_direction = unvisited_directions[0][0]
                else:
                    selected_direction = visited_directions[0][0]
            else:
                # Case 2: There are unvisited paths to go
                if len(unvisited_directions) != 0:
                    self._visit_map[current_cell[0], current_cell[1]] = _VISITED
                    selected_direction = self._heuristic.select(unvisited_directions)[0]
                else:
                    # Case 3: We can backtrace our path
                    backtrace_direction = None
                    for direction, direction_cell in visited_directions:
                        if (previous_cell == direction_cell).all():
                            backtrace_direction = direction
                            break
                    if backtrace_direction is not None:
                        selected_direction = backtrace_direction
                        self._visit_map[current_cell[0], current_cell[1]] = _BLOCKED
                    else:
                        # Case 4: we can not backtrace
                        selected_direction = self._heuristic.select(visited_directions)[0]
                        self._visit_map[current_cell[0], current_cell[1]] += _VISITED

            previous_cell = current_cell
            self._move_robot(selected_direction)
        
    def _get_directions(self, sensors: SensorsReading) -> Tuple[bool,
                                                                List[Tuple[PossibleDirection, np.ndarray]],
                                                                List[Tuple[PossibleDirection, np.ndarray]],
                                                                int]:
        front_wall, right_wall, back_wall, left_wall = detect_walls(sensors, self._walls_threshold)
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
            raise RuntimeError("Happened something that never could happened")

    def _is_in_center(self, cell: np.ndarray) -> bool:
        cell = (int(cell[0]), int(cell[1]))
        return cell in ((7, 7),
                        (7, 8),
                        (8, 7),
                        (8, 8))
