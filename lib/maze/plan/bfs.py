import numpy as np

from collections import deque
from lib.maze.plan.algo_base import AbstractPlanner
from lib.entities.state import Orientation
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import AbstractSensorsReader
from lib.maze.common import WALLS_DICT


class BFSPlanner(AbstractPlanner):
    """Breadth-first search planner implementation.
    """

    def __init__(self, 
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 maze: np.ndarray):
        super(BFSPlanner, self).__init__(controller, sensors_reader, maze)

    def find_path(self, start_cell: np.ndarray, target_cell: np.ndarray) -> np.ndarray:
        """Finds a path -s equence of cell coordinates - from start_cell to target_cell.

        Args:
            start_cell (np.ndarray): Start of the path
            target_cell (np.ndarray): Target cell of the path
        """
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
