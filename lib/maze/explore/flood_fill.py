import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from typing import List, Tuple, Optional
from collections import deque
from lib.maze.explore.algo_base import AbstractMazeExplorer
from lib.entities.sensors import SensorsReading
from lib.entities.state import Orientation
from lib.entities.cell import Cell
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import AbstractSensorsReader
from lib.maze.common import MAZE_SIDE, PossibleDirection
from lib.maze.util import detect_walls, update_cell
from lib.maze.explore.heuristic import AbstractJunctionHeuristic


class FloodFillMap:

    def __init__(self,
                 prebuild_walls: Optional[np.ndarray] = None):
        self._inited = False
        if prebuild_walls is None:
            self._walls = self._build_initial_walls()
        else:
            self._walls = prebuild_walls.copy()
        self._cost_map = self._build_cost_map(self._walls)

    def _update_plot_walls(self, walls):
        for i in range(16):
            for j in range(16):
                north, east, south, west = walls[i, j]
                if north:
                    self._matrix_ax.add_patch(patches.Rectangle(
                        (j-0.5, i-0.5), 1, 0, edgecolor='red', lw=2))
                if south:
                    self._matrix_ax.add_patch(patches.Rectangle(
                        (j-0.5, i+0.5), 1, 0, edgecolor='red', lw=2))

                if east:
                    self._matrix_ax.add_patch(patches.Rectangle(
                        (j+0.5, i-0.5), 0, 1, edgecolor='red', lw=2))
                if west:
                    self._matrix_ax.add_patch(patches.Rectangle(
                        (j-0.5, i-0.5), 0, 1, edgecolor='red', lw=2))

    def _update_cells_text(self, matrix):
        for i in range(16):
            for j in range(16):
                self._matrix_ax.text(i, j, f'{(matrix[15-j, i]).astype(int)}',
                                     ha='center', va='center', fontsize=8,
                                     color='white')

    def _update_colored_matrix(self, matrix=np.zeros((16, 16))):
        if self._matrix_fig is None:
            self._matrix_fig, self._matrix_ax = plt.subplots()
            plt.ion()
            plt.show()

        self._matrix_ax.clear()
        self._matrix_display = self._matrix_ax.imshow(
            matrix, cmap='winter', interpolation='nearest')
        self._update_plot_walls(self._walls)
        self._update_cells_text(matrix)

        self._matrix_fig.canvas.draw()
        plt.pause(0.01)

    def get_walls(self) -> np.ndarray:
        return self._walls.copy()

    def add_walls(self, cell: np.ndarray, walls: Tuple[bool, bool, bool, bool]):
        self._walls[cell[0], cell[1], :] = np.array(walls)

    def rebuild_cost_map(self):
        self._cost_map = self._build_cost_map(self._walls)

    def cost_at(self, cell: np.ndarray) -> int:
        return self._cost_map[cell[0], cell[1]]

    def _build_initial_walls(self) -> np.ndarray:
        # row, col, (north, east, south, west)
        walls = np.zeros((MAZE_SIDE, MAZE_SIDE, 4), dtype=bool)
        walls[:, 0, 3] = True  # West wall
        walls[:, 15, 1] = True  # East wall
        walls[0, :, 0] = True  # North wall
        walls[15, :, 2] = True  # South wall
        return walls

    def _build_cost_map(self, walls: np.ndarray) -> np.ndarray:
        # Initialize central cells and blank cell
        cost_map = np.ones((MAZE_SIDE, MAZE_SIDE)) * -1
        visit_map = np.zeros((MAZE_SIDE, MAZE_SIDE), dtype=bool)
        cost_map[7, 7] = 0
        visit_map[7, 7] = True

        cell_queue = deque()
        cell_queue.append((7, 7))

        while len(cell_queue) != 0:
            cell = cell_queue.popleft()
            neighbours = self._get_cell_neighbours(cell, walls, cost_map)
            for nb in neighbours:
                if not visit_map[nb[0], nb[1]]:
                    visit_map[nb[0], nb[1]] = True
                    cost_map[nb[0], nb[1]] = cost_map[cell[0], cell[1]] + 1
                    cell_queue.append(nb)

        return cost_map

    def _get_cell_neighbours(self,
                             cell: Tuple[int, int],
                             walls: np.ndarray,
                             cost_map: np.ndarray) -> List[Tuple[int, int]]:
        result = []
        displacements = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        oppostie_directions = [2, 3, 0, 1]
        for orientation, displacement in enumerate(displacements):
            opposite_orientation = oppostie_directions[orientation]
            neighbour = (cell[0] + displacement[0], cell[1] + displacement[1])
            if (0 <= neighbour[0] < MAZE_SIDE) and (0 <= neighbour[1] < MAZE_SIDE):
                if not walls[cell[0], cell[1], orientation] and not walls[neighbour[0], neighbour[1], opposite_orientation]:
                    if cost_map[neighbour[0], neighbour[1]] == -1:
                        result.append((int(neighbour[0]), int(neighbour[1])))
        return result


class FloodFillExplorer(AbstractMazeExplorer):
    """Flood Fill algorithm implementation for the unknown maze exploration.
    """

    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader,
                 walls_threshold: float,
                 yaw_eps: float,
                 semifinal_mode: bool,
                 prebuild_walls: Optional[np.ndarray] = None):
        """Instanitates the explorer.

        Args:
            controller (AbstractRobotController): Robot controller
            sensors_reader (AbstractSensorsReader): Sensors reader
            walls_threshold (float): Minimal distance to the wall at 
                                     which wall is considered as detected 
                                     and movement is forbidden in corresponding direction
            yaw_eps (float): Tolerance of the yaw orientation quadrant
        """
        super(FloodFillExplorer, self).__init__()
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._walls_threshold = walls_threshold
        self._yaw_eps = yaw_eps
        self._semifinal_mode = semifinal_mode

        self._flood_fill_map = FloodFillMap(prebuild_walls)

        self._maze_map = np.ones((MAZE_SIDE, MAZE_SIDE), dtype=int) * -1

    def run(self) -> np.ndarray:
        """Run the exploration (mapping) process.

        Returns:
            np.ndarray: built map of the maze
        """
        while True:
            sensors_reading = self._sensors_reader.get_reading()
            current_cell = self._controller.odometry.cell
            update_cell(current_cell, self._maze_map, sensors_reading,
                        self._walls_threshold, self._yaw_eps, self._semifinal_mode)

            if self._is_in_center(current_cell):
                return self._maze_map.copy(), {"walls": self._flood_fill_map.get_walls()}

            if (self._maze_map != -1).all():
                return self._maze_map.copy(), {"walls": self._flood_fill_map.get_walls()}

            walls = detect_walls(sensors_reading, self._walls_threshold)
            self._add_walls(walls)

            direction, feasible = self._get_directions(walls)

            self._flood_fill_map.rebuild_cost_map()
            
            self._move_robot(direction)

    def _get_directions(self, walls: Tuple[bool, bool, bool, bool]) -> Tuple[PossibleDirection, bool]:
        front_wall, right_wall, back_wall, left_wall = walls
        can_front = not front_wall
        can_right = not right_wall
        can_back = not back_wall
        can_left = not left_wall

        odom = self._controller.odometry
        direction = None
        min_cost = np.inf
        feasible = False

        next_cell = odom.move_forward().cell
        if self._check_next_cell(next_cell):
            cost = self._flood_fill_map.cost_at(next_cell)
            if cost < min_cost and can_front:
                min_cost = cost
                direction = PossibleDirection.FRONT
                feasible = can_front

        next_cell = odom.rotatate_right().move_forward().cell
        if self._check_next_cell(next_cell):
            cost = self._flood_fill_map.cost_at(next_cell)
            if cost < min_cost and can_right:
                min_cost = cost
                direction = PossibleDirection.RIGHT
                feasible = can_right

        next_cell = odom.rotate_left().move_forward().cell
        if self._check_next_cell(next_cell):
            cost = self._flood_fill_map.cost_at(next_cell)
            if cost < min_cost and can_left:
                min_cost = cost
                direction = PossibleDirection.LEFT
                feasible = can_left

        next_cell = odom.move_backward().cell
        if self._check_next_cell(next_cell):
            cost = self._flood_fill_map.cost_at(next_cell)
            if cost < min_cost and can_back:
                min_cost = cost
                direction = PossibleDirection.BACK
                feasible = can_back

        return direction, feasible

    def _check_next_cell(self, next_cell: np.ndarray) -> bool:
        if (next_cell >= MAZE_SIDE).any() or (next_cell < 0).any():
            return False
        return True

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
            raise RuntimeError(f"Wrong robot direction {direction}. Robot is possibly stuck or odometry is innacurate. Current cell is {self._controller.odometry.cell}")

    def _is_in_center(self, cell: np.ndarray) -> bool:
        cell = (int(cell[0]), int(cell[1]))
        return cell in ((7, 7),
                        (7, 8),
                        (8, 7),
                        (8, 8))

    def _add_walls(self, walls: Tuple[bool, bool, bool, bool]):
        front_wall, right_wall, back_wall, left_wall = walls
        wall_key = np.array([front_wall, right_wall, back_wall, left_wall])
        cell = self._controller.odometry.cell
        orientation = self._controller.odometry.orientation
        if orientation == Orientation.NORTH:
            roll_step = 0
        elif orientation == Orientation.EAST:
            roll_step = 1
        elif orientation == Orientation.SOUTH:
            roll_step = 2
        elif orientation == Orientation.WEST:
            roll_step = 3
        wall_key = np.roll(wall_key, roll_step)
        wall_key = tuple(wall_key.tolist())
        # print(f'added {wall_key} walls, {"North" if wall_key[0] else ""}, {"East" if wall_key[1] else ""}, {"South" if wall_key[2] else ""}, {"West" if wall_key[3] else ""} ')
        self._flood_fill_map.add_walls(cell, wall_key)

    def _select_direction(self, possible_directions: List[Tuple[PossibleDirection, np.ndarray]]) -> PossibleDirection:
        min_value = np.inf
        selected_direction = None
        costs = []
        for direction, cell in possible_directions:
            cost = self._flood_fill_map.cost_at(cell)
            # print(cost)
            if cost < min_value:
                min_value = cost
                selected_direction = direction
        # print(f'{selected_direction} from {possible_directions}')
        # print(costs),
        # print(f'min {min_value}')
        # print("===")
        return selected_direction
