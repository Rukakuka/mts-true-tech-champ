import enum
import requests
import numpy as np
import time
from pynput import keyboard

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


API_TOKEN = "73b5db4b-735a-473c-9378-83fbcf97b167cd82fd17-a895-488a-8cfb-e3c4a321c1b1"


def get_maze():
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
    _PWM_URL = "http://localhost:8801/api/v1/robot-motors/move"

    def __init__(self, api_token: str, delay: float = 0.1) -> None:
        self._token = api_token
        self._current_cell = (15, 0)
        self._current_orientation = Orientation.FORWARD
        reading = self.read_sensors()
        self._last_offset = (reading.down_x_offset, reading.down_y_offset)
        self._delay = delay
        self._x = 0
        self._y = 15
        self._yaw = 0

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
        self._current_cell = (
            self._current_cell[0] + d1, self._current_cell[1] + d2)
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
        self._current_cell = (
            self._current_cell[0] + d1, self._current_cell[1] + d2)
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

    def _movement_request(self, direction: str) -> bool:
        url = f"{RoboController._BASE_URL}/{direction}?token={self._token}"
        response = requests.post(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        time.sleep(self._delay)
        return True

    def send_matrix(self, matrix) -> bool:
        url = f"{RoboController._MATRIX_URL}?token={self._token}"
        try:
            response = requests.post(url, json=matrix)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Received status code {response.status_code}")
            return True
        finally:
            time.sleep(self._delay)

    def _pwm_request(self, pwm_l: int, pwm_r: int, time_l: float, time_r: float) -> bool:
        url = f"{RoboController._PWM_URL}?l={pwm_l}&l_time={time_l}&r={pwm_r}&r_time={time_r}"
        response = requests.post(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        return True

    def _angle_norm(self, angle):
        if angle > np.pi/2:
            return angle - np.pi
        elif angle < -np.pi/2:
            return angle + np.pi
        return angle

    def calculate_corrections(self, pwm_base, gt, actual, next_cell):
        # Rotate to compensate d yaw
        error = gt - actual
        error[2][0] = self._angle_norm(self._angle_norm(gt[2][0]) - self._angle_norm(actual[2][0]))
        print(f'angle error {error[2][0]}')
        next_cell = np.array([[next_cell[0]], [next_cell[1]], gt[2]])
        step = next_cell - gt

        k = 10

        a = np.deg2rad(error[2][0]) * k
        matrix = np.array([[np.cos(a), -np.sin(a)],
                           [np.sin(a), np.cos(a)]])

        propagated = np.array(
            [[step[0][0] + error[0][0]], [step[1][0] + error[1][0]], [0]])

        pwm = np.array([[pwm_base], [pwm_base]])
        rotated_pwm = np.dot(matrix.T, pwm)
        factor = np.max(rotated_pwm)/pwm_base
        rotated_pwm = rotated_pwm / factor

        # # Rotate to compensate d y
        # rotated_error = np.dot(matrix.T, np.array([[error[0][0]], [error[1][0]]]))

        # theta_1_rad = np.abs(np.arctan(propagated[0][0] / propagated[1][0]))
        # theta_2_rad = np.pi/2 - theta_1_rad

        # matrix2 = np.array([[np.cos(-theta_1_rad), -np.sin(-theta_1_rad)],
        #                     [np.sin(-theta_1_rad), np.cos(-theta_1_rad)]])

        # rotated_pwm = np.dot(matrix2, rotated_pwm)
        # factor = np.max(rotated_pwm)/255
        # rotated_pwm /= (1+factor)*k

        # Scale to compensate d x

        k = 0.4

        prop_len = np.linalg.norm([propagated[0][0], propagated[1][0]])
        step_len = np.linalg.norm([step[0][0], step[1][0]])

        print(f'{propagated=}')
        print(f'{step=}')
        print(f'{prop_len=}')
        print(f'{step_len=}')

        factor = ((prop_len/step_len - 1) * k ) + 1
        rotated_pwm = rotated_pwm * (np.max(rotated_pwm * factor)/pwm_base)

        return rotated_pwm

    def normalize_angle(self, angle):
        return (angle + 360) % 360

    def get_orientation(self):
        normalized_yaw = self.normalize_angle(self._yaw)
        # print(normalized_yaw)
        if 315 <= normalized_yaw < 360 or 0 <= normalized_yaw < 45:
            return 'N'
        elif 45 <= normalized_yaw < 135:
            return 'E'
        elif 135 <= normalized_yaw < 225:
            return 'S'
        elif 225 <= normalized_yaw < 315:
            return 'W'
        else:
            raise ValueError

    yaw_lookup = {
        'N': 0,
        'E': np.pi/2,
        'S': -np.pi,
        'W': -np.pi/2
    }

    update_pos_lookup = {
        'N': [0, -1],
        'E': [1, 0],
        'S': [0, 1],
        'W': [-1, 0]
    }

    def discrete_forward(self):
        sens = self.read_sensors()
        self._yaw = sens.rotation_yaw

        actual = np.array([[sens.down_x_offset], [sens.down_y_offset], [
                          np.deg2rad(sens.rotation_yaw)]])
        maze = get_maze()

        gt_x = -maze[self._x][self._y][1]
        gt_y = -maze[self._x][self._y][0]

        gt_yaw = self.yaw_lookup[self.get_orientation()]

        gt = np.array([[gt_x], [gt_y], [gt_yaw]])
        next_cell = None

        orientation = self.get_orientation()
        try:
            if (orientation == 'N'):
                next_cell = maze[self._y-1][self._x]
            elif (orientation == 'E'):
                next_cell = maze[self._y][self._x+1]
            elif (orientation == 'S'):
                next_cell = maze[self._y+1][self._x]
            elif (orientation == 'W'):
                next_cell = maze[self._y][self._x-1]
        except IndexError:
            next_cell = gt

        print(f'\nactual {[f"{x.item():5.3f}" for x in actual]}')
        print(f'gt {[f"{x.item():5.3f}" for x in gt]}')
        print(f'next {[f"{x.item():5.3f}" for x in next_cell]}')
        
        pwm_base = 200
        # throttle_time_fwd = 0.51    # 255
        # braking_time_fwd = 0.31     # 255
        throttle_time_fwd = 1.1    # 200
        braking_time_fwd = 0.4     # 200
        # throttle_time_fwd = 0.92    # 225
        # braking_time_fwd = 0.37    # 225
        
        pwm = self.calculate_corrections(pwm_base, gt, actual, next_cell)

        pwm_l = pwm[0][0].astype(int)
        pwm_r = pwm[1][0].astype(int)

        print(pwm_l, pwm_r)


        self._pwm_request(pwm_l, pwm_r, throttle_time_fwd, throttle_time_fwd)
        time.sleep(max(throttle_time_fwd, 0.7))
        self._pwm_request(-pwm_l, -pwm_r, braking_time_fwd, braking_time_fwd)
        time.sleep(max(braking_time_fwd, 0.5))
        self._pwm_request(0, 0, 0.3, 0.3)
        time.sleep(0.3)

        xy = self.update_pos_lookup[self.get_orientation()]
        x = self._x + xy[0]
        y = self._y + xy[1]
        self._x = (x if x >= 0 and x <= 15 else self._x)
        self._y = (y if y >= 0 and y <= 15 else self._y)

    def discrete_backward(self):
        throttle_time_rwd = 0.44
        braking_time_rwd = 0.38
        self._pwm_request(-255, -255, throttle_time_rwd, throttle_time_rwd)
        time.sleep(max(throttle_time_rwd, 0.7))
        self._pwm_request(255, 255, braking_time_rwd, braking_time_rwd)
        time.sleep(max(braking_time_rwd, 0.5))
        self._pwm_request(0, 0, 0.3, 0.3)
        time.sleep(0.3)

    def _discrete_rotate(self, dir):
        throttle_time_rotate = 1.01
        self._pwm_request(-255 * dir, 255 * dir,
                          throttle_time_rotate, throttle_time_rotate)
        time.sleep(max(throttle_time_rotate, 0.7))
        self._pwm_request(0, 0, 0.3, 0.3)
        time.sleep(0.3)

    def discrete_right(self):
        self._discrete_rotate(-1)

    def discrete_left(self):
        self._discrete_rotate(1)


rc = RoboController(API_TOKEN)

if __name__ == '__main__':
    def on_press(key):
        try:
            if key == keyboard.Key.up:
                rc.discrete_forward()
            elif key == keyboard.Key.down:
                rc.discrete_backward()
            elif key == keyboard.Key.right:
                rc.discrete_right()
            elif key == keyboard.Key.left:
                rc.discrete_left()
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
