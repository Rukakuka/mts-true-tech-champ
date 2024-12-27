import time
import numpy as np

from typing import Optional
from lib.robot.base import AbstractRobotController
from lib.api.client import ApiClient
from lib.entities.state import Orientation
from lib.robot.odometer import OdometryState


_MAZE_CELLS = [[[1253.94, -1254.48],
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


class PD:
    """PD controller implementation.
    """
    def __init__(self, 
                 kp: float, 
                 kd: float, 
                 max_limit: Optional[float] = None, 
                 min_limit: Optional[float] = None):
        self._kp = kp
        self._kd = kd
        self._max_limit = max_limit
        self._min_limit = min_limit
        self._last_time = None
        self._previous_error = None
        
    def _max_clip(self, value):
            return np.clip(value, -self._max_limit, self._max_limit)

    def __call__(self, error):
        dt = 0 if self._last_time is None else (time.time() - self._last_time)
        self._last_time = time.time()
        derivative = 0 if self._previous_error is None else (error - self._previous_error) / dt
        self._previous_error = error

        output = (self._kp * error) + (self._kd * derivative)
        
        if self._max_limit is not None:
            output = self._max_clip(output)

        if self._min_limit is not None:
            output = np.sign(output + 0.00001) * max(np.abs(output), self._min_limit)

        return output


class PdPwmRobotController(AbstractRobotController):

    def __init__(self, 
                 api_client: ApiClient, 
                 delay: Optional[float],
                 kp_linear: float, 
                 kd_linear: float, 
                 kp_angular: float, 
                 kd_angular: float, 
                 distance_offset: float, 
                 angle_offset: float, 
                 success_rate_linear: int,
                 success_rate_angular: int,
                 lenght: float, 
                 radius: float, 
                 pwm_max: int, 
                 pwm_time: float):
        super(PdPwmRobotController, self).__init__()
        self._api_client = api_client
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.distance_offset = distance_offset
        self.angle_offset = angle_offset
        self.maze = np.array(_MAZE_CELLS)
        self.orientation_dict = {0: Orientation.NORTH, 
                                 90: Orientation.EAST, 
                                 180: Orientation.SOUTH, 
                                 -90: Orientation.WEST}
        self.lenght = lenght
        self.radius = radius
        self.pwm_max = pwm_max
        self.pwm_time = pwm_time
        self._success_rate_linear = success_rate_linear
        self._success_rate_angular = success_rate_angular
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
        self._shift_move(n)
        self._do_delay()

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        self._shift_move(-n)
        self._do_delay()

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        self._shift_rotate(-n)
        self._do_delay()

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        self._shift_rotate(n)
        self._do_delay()

    def _shift_move(self, shift: int):
        reading = self._api_client.read_sensors()
        current_angle = reading.rotation_yaw
        current_state = np.array([reading.down_x_offset, reading.down_y_offset])
        current_position = self._odometry.cell
        position = current_position.copy()

        index, inverse = self._nearest_direction(current_angle)

        position[index] += inverse * shift if index else -inverse * shift
        target_coordinate = self.maze[position[0], position[1], index]

        if shift > 0:
            pd = PD(self.kp_linear, self.kd_linear, max_limit=300, min_limit=50)
        else:
             pd = PD(self.kp_linear, self.kd_linear*1.8, max_limit=300, min_limit=50)
        error = target_coordinate - current_state[index]

        success_counter = 0
        while success_counter < self._success_rate_linear:
            reading = self._api_client.read_sensors()
            current_state = np.array([reading.down_x_offset, reading.down_y_offset])
            error = target_coordinate - current_state[index]
            speed = pd(error)

            self._move_pwm(np.array([inverse * speed, 0]))
            if abs(error) < self.distance_offset:
                success_counter += 1

        self._move_pwm(np.array([0, 0]))

        self._odometry = OdometryState(cell=position,
                                       orientation=self._odometry.orientation)
        # print(self._odometry.cell, self._odometry.orientation)

    def _shift_rotate(self, shift: int):
        shift = self._nearest_angle(shift * 90)
        reading = self._api_client.read_sensors()
        initial_angle = reading.rotation_yaw
        target_angle = self._nearest_angle(initial_angle + shift)
        # self._current_orientation = self.orientation_dict[target_angle]

        pd = PD(self.kp_angular, self.kd_angular, min_limit=55, max_limit=500)
        error = self._angle_norm(target_angle - reading.rotation_yaw)

        success_counter = 0
        while success_counter < self._success_rate_angular:
            reading = self._api_client.read_sensors()
            error = self._angle_norm(target_angle - reading.rotation_yaw)
            speed = pd(error)
            self._move_pwm(np.array([0, speed]))

            if abs(error) < self.angle_offset:
                success_counter += 1

        self._move_pwm(np.array([0, 0]))

        self._odometry = OdometryState(cell=self._odometry.cell,
                                       orientation=self.orientation_dict[target_angle])
        # print(self._odometry.cell, self._odometry.orientation)

    def _do_delay(self):
        if self._delay is not None:
            time.sleep(self._delay)

    def _angle_norm(self, angle):
        if angle > 180:
            return angle - 360
        elif angle < -180:
            return angle + 360
        return angle
    
    def _angle_distance(self, a, b):
            return min((a - b) % 360, (b - a) % 360)

    def _nearest_angle(self, angle):
        target_angles = [0, 90, 180, -90]
        nearest = min(target_angles, key=lambda x: self._angle_distance(angle, x))
        return nearest
    
    def _nearest_direction(self, angle):
        nearest_angle = self._nearest_angle(angle)
        if nearest_angle in (90, -90):
            index = 1
        else:
            index = 0
        
        if nearest_angle in (0, 90):
            direction = 1
        else: 
            direction = -1
        return index, direction

    def _move_pwm(self, vel: np.ndarray):
        vel = np.array(vel)
        pwm = self._vel2pwm(vel)
        self._api_client.pwm(pwm_l=pwm[1],
                             pwm_l_dt=self.pwm_time,
                             pwm_r=pwm[0],
                             pwm_r_dt=self.pwm_time)

    def _vel2pwm(self, vel: np.ndarray):
        V, w = vel
        coeff = np.array((-1,1))
        pwm = (V + coeff * w * self.lenght) / (2 * self.radius)
        pwm = np.clip(pwm, -self.pwm_max, self.pwm_max)
        # (r,l)
        pwm = pwm.astype(int)
        return pwm
