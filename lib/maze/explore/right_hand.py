import numpy as np

from lib.maze.explore.algo_base import AbstractMazeExplorer
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import AbstractSensorsReader


MAZE_SIZE = 16
WALL_THRESHOLD = 65


class RightHandExplorer(AbstractMazeExplorer):
    """Right hand algorithm implementation for the unknown maze exploration.
    """

    def __init__(self,
                 controller: AbstractRobotController,
                 sensors_reader: AbstractSensorsReader
                 ):

        super(RightHandExplorer, self).__init__()
        self._controller = controller
        self._sensors_reader = sensors_reader
        self._steps = 0
        self._maze = np.full((MAZE_SIZE, MAZE_SIZE), -1, dtype=int)
        self._max_steps = MAZE_SIZE * MAZE_SIZE * 4
        self._current_position = [MAZE_SIZE - 1, 0]

    def _normalize_angle(self, angle):
        return (angle + 360) % 360

    def _get_orientation(self, yaw):
        normalized_yaw = self._normalize_angle(yaw)
        if 315 <= normalized_yaw < 360 or 0 <= normalized_yaw < 45:
            return 'N'
        elif 45 <= normalized_yaw < 135:
            return 'E'
        elif 135 <= normalized_yaw < 225:
            return 'S'
        elif 225 <= normalized_yaw < 315:
            return 'W'

    def _update_position(self, move, orientation):
        new_position = self._current_position.copy()
        if orientation == 'N' and move == 'right':
            new_position[1] += 1
        elif orientation == 'N' and move == 'forward':
            new_position[0] -= 1

        elif orientation == 'E' and move == 'right':
            new_position[0] += 1
        elif orientation == 'E' and move == 'forward':
            new_position[1] += 1

        elif orientation == 'S' and move == 'right':
            new_position[1] -= 1
        elif orientation == 'S' and move == 'forward':
            new_position[0] += 1

        elif orientation == 'W' and move == 'right':
            new_position[0] -= 1
        elif orientation == 'W' and move == 'forward':
            new_position[1] -= 1

        if 0 <= new_position[0] < MAZE_SIZE and 0 <= new_position[1] < MAZE_SIZE:
            self._current_position = new_position.copy()
        else:
            print(
                f"Предупреждение: попытка выйти за границы лабиринта. Текущая позиция: {self._current_position}")

    def _detect_walls(self, sensor_data, orientation):
        front = sensor_data['front_distance'] < WALL_THRESHOLD
        right = sensor_data['right_side_distance'] < WALL_THRESHOLD
        left = sensor_data['left_side_distance'] < WALL_THRESHOLD
        back = sensor_data['back_distance'] < WALL_THRESHOLD

        walls = {
            'N': 0,
            'E': 0,
            'S': 0,
            'W': 0
        }

        if orientation == 'N':
            walls['N'], walls['E'], walls['S'], walls['W'] = front, right, back, left
        elif orientation == 'E':
            walls['N'], walls['E'], walls['S'], walls['W'] = left, front, right, back
        elif orientation == 'S':
            walls['N'], walls['E'], walls['S'], walls['W'] = back, left, front, right
        elif orientation == 'W':
            walls['N'], walls['E'], walls['S'], walls['W'] = right, back, left, front

        return walls

    def _calculate_cell_value(self, walls):
        if not walls['W'] and not walls['N'] and not walls['E'] and not walls['S']:
            return 0
        elif walls['W'] and not walls['N'] and not walls['E'] and not walls['S']:
            return 1
        elif not walls['W'] and walls['N'] and not walls['E'] and not walls['S']:
            return 2
        elif not walls['W'] and not walls['N'] and walls['E'] and not walls['S']:
            return 3
        elif not walls['W'] and not walls['N'] and not walls['E'] and walls['S']:
            return 4
        elif walls['W'] and not walls['N'] and not walls['E'] and walls['S']:
            return 5
        elif not walls['W'] and not walls['N'] and walls['E'] and walls['S']:
            return 6
        elif not walls['W'] and walls['N'] and walls['E'] and not walls['S']:
            return 7
        elif walls['W'] and walls['N'] and not walls['E'] and not walls['S']:
            return 8
        elif walls['W'] and not walls['N'] and walls['E'] and not walls['S']:
            return 9
        elif not walls['W'] and walls['N'] and not walls['E'] and walls['S']:
            return 10
        elif not walls['W'] and walls['N'] and walls['E'] and walls['S']:
            return 11
        elif walls['W'] and walls['N'] and walls['E'] and not walls['S']:
            return 12
        elif walls['W'] and walls['N'] and not walls['E'] and walls['S']:
            return 13
        elif walls['W'] and not walls['N'] and walls['E'] and walls['S']:
            return 14
        elif walls['W'] and walls['N'] and walls['E'] and walls['S']:
            return 15

    def _update_maze(self, walls):

        cell_value = self._calculate_cell_value(walls)
        if 0 <= self._current_position[0] < MAZE_SIZE and 0 <= self._current_position[1] < MAZE_SIZE:
            self._maze[self._current_position[0],
                       self._current_position[1]] = cell_value
        else:
            print(
                f"Ошибка: попытка обновить ячейку за пределами лабиринта. Позиция: {self._current_position}")

    def _move_robot(self, sensor_data):
        # Логика обхода по правилу правой руки
        if sensor_data['right_side_distance'] > 65:  # Если справа свободно
            self._controller.rotate_right()
            self._controller.move_forward()
            return "right"
        elif sensor_data['front_distance'] > 65:  # Если впереди свободно
            self._controller.move_forward()
            return "forward"
        else:
            # Поворачиваем налево если впереди и справа есть стены
            self._controller.rotate_left()
            return "left"

    def run(self) -> np.ndarray:
        while -1 in self._maze and self._steps < self._max_steps:
            sensor_data = self._sensors_reader.get_reading().as_dict()
            orientation = self._get_orientation(sensor_data['rotation_yaw'])
            walls = self._detect_walls(sensor_data, orientation)
            self._update_maze(walls)
            move = self._move_robot(sensor_data)
            self._update_position(move, orientation)

            self._steps += 1

        if self._steps >= self._max_steps:
            print("Достигнуто максимальное количество шагов. Возможно, робот зациклился.")
        else:
            print("Лабиринт полностью исследован!")

        return self._maze.copy(), {}
