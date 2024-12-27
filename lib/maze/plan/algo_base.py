import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import AbstractSensorsReader
from lib.entities.state import Orientation


class AbstractPlanner(ABC):
    """Base class for implementing planning algorithms.
    """

    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 maze: np.ndarray):
        """Instantiates the planner.

        Args:
            controller (AbstractRobotController): Controller to move the robot
            sensors_reader (AbstractSensorsReader): Sensors reader
            maze (np.ndarray): Pre-build maze map
        """
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._maze = maze.copy()
        self._center_entrance = self._find_center_entrance()

    @abstractmethod
    def find_path(self, start_cell: np.ndarray, target_cell: np.ndarray):
        """Finds a path -s equence of cell coordinates - from start_cell to target_cell.

        Args:
            start_cell (np.ndarray): Start of the path
            target_cell (np.ndarray): Target cell of the path
        """
        pass

    def run(self, precalculated_plan=None):
        """Runs the planning pipeline: find path from current cell to the maze center
        and then execute the found path, or directrly use precalculated one.
        """
        start_cell = self._controller.odometry.cell
        plan = self.find_path(start_cell, self._center_entrance) if precalculated_plan is None else precalculated_plan
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
            
            # Determine the most efficient way to turn
            current_index = current_orientation.value
            target_index = target_orientation.value
            
            turn_difference = (target_index - current_index + 4) % 4
            
            if turn_difference == 0:
                # Already facing the right direction, no need to turn
                pass
            elif turn_difference == 1:
                self._controller.rotate_right()
            elif turn_difference == 2:
                # It's faster to move backward
                pass
            else:
                self._controller.rotate_left()

            current_orientation = target_orientation
            
            # Move in the correct direction
            if turn_difference == 2:
                self._controller.move_backward()
            else:
                self._controller.move_forward()

    def _find_center_entrance(self) -> np.ndarray:
        central_cells = [(7,7), (7,8), (8,7),  (8,8)]
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
        for central_cell in central_cells:
            if self._is_entry(central_cell, dict_enter):
                return np.array(central_cell)

    def _is_entry(self, 
                  cell_candidate: Tuple[int, int],
                  dict_enter: Dict[Tuple[int, int], List[int]]) -> bool:
        x,y = cell_candidate
        neighbours_list = [(x-1,y), (x+1,y), (x,y-1), (x, y+1)]
        neighbour_list_types = []
        for cell in neighbours_list:
            x, y = cell
            neighbour_list_types.append(self._maze[x,y])

        neighbours_list = np.array(neighbours_list)
        neighbour_list_types = np.array(neighbour_list_types)

        neighbours_list_filtered = neighbours_list[neighbour_list_types != -1]
        neighbours_list_filtered_tuples = [tuple(x) for x in neighbours_list_filtered]
        for neighbour in neighbours_list_filtered_tuples:
            x,y = neighbour
            value = self._maze[x,y,]
            if neighbour in dict_enter and value in dict_enter[neighbour]:
                return True
