import time
import numpy as np
import multiprocessing
import queue

from typing import Optional
from lib.robot.base import AbstractRobotController
from lib.api.client import ApiClient
from lib.entities.state import Orientation
from lib.robot.odometer import OdometryState
from lib.maze.util import get_ground_truth_cell_coordinates


class DiscreteAsyncPollController(AbstractRobotController):

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
        super(DiscreteAsyncPollController, self).__init__()
        self._api_client = api_client
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.distance_offset = distance_offset
        self.angle_offset = angle_offset
        self.maze = np.array(get_ground_truth_cell_coordinates())
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
        self._poll_queue = multiprocessing.Queue(maxsize=1)
        self._poller_process = multiprocessing.Process(
            target=self._poller, args=())
        self._poller_process.start()
        self._x = 0
        self._y = 15
        self._yaw = 0

    @property
    def odometry(self) -> OdometryState:
        """Current cell position and orientation of the robot.
        """
        return self._odometry

    def move_forward(self, n: int = 1):
        """Moves robot n cells forward (in local frame).
        """
        for i in range(n):
            self._discrete_forward()
            self._do_delay()

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        for i in range(n):
            self._discrete_backward()
            self._do_delay()

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        for i in range(n):
            self._discrete_left()
            self._do_delay()

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        for i in range(n):
            self._discrete_right()
            self._do_delay()

    def _poller(self):
        cnt = 0
        t = time.monotonic()
        while True:
            sensors = self._api_client.read_sensors()
            try:
                self._poll_queue.put(sensors, block=False)
            except queue.Full:
                try:
                    _ = self._poll_queue.get(block=False)
                except queue.Empty:
                    pass

            cnt += 1
            if (cnt % 100 == 0):
                dt = time.monotonic() - t
                print(
                    f'Async sensor poll rate is {cnt/dt:.3f} Hz')

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

    def _discrete_forward(self):
        sens = self._poll_queue.get()
        self._yaw = sens.rotation_yaw

        actual = np.array([[sens.down_x_offset], [sens.down_y_offset], [
                          np.deg2rad(sens.rotation_yaw)]])
        maze = get_ground_truth_cell_coordinates()

        gt_x = maze[self._y][self._x][0]
        gt_y = maze[self._y][self._x][1]

        gt_yaw = self.yaw_lookup[self.get_orientation()]

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

        pwm_base = 220
        throttle_time_fwd = 0.51    # 255
        braking_time_fwd = 0.31     # 255

        pwm = self._calculate_corrections(pwm_base, gt, actual, next_cell)

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

    def _discrete_backward(self):
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

    def _discrete_right(self):
        self._discrete_rotate(-1)

    def _discrete_left(self):
        self._discrete_rotate(1)

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

    def _pwm_request(self, pwm_l: int, pwm_r: int, time_l: float, time_r: float) -> bool:
        self._api_client.pwm(pwm_l=pwm_l,
                             pwm_l_dt=time_l,
                             pwm_r=pwm_r,
                             pwm_r_dt=time_r)

    def _angle_norm(self, angle):
        if angle > np.pi/2:
            return angle - np.pi
        elif angle < -np.pi/2:
            return angle + np.pi
        return angle

    def _calculate_corrections(self, pwm_base, gt, actual, next_cell):
        # Rotate to compensate d yaw
        error = gt - actual
        error[2][0] = self._angle_norm(self._angle_norm(
            gt[2][0]) - self._angle_norm(actual[2][0]))
        print(f'angle error {error[2][0]}')
        next_cell = np.array([[next_cell[0]], [next_cell[1]], gt[2]])
        step = next_cell - gt

        k = 12

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

        k = 1.0

        prop_len = np.linalg.norm([propagated[0][0], propagated[1][0]])
        step_len = np.linalg.norm([step[0][0], step[1][0]])

        print(f'{propagated=}')
        print(f'{step=}')
        print(f'{prop_len=}')
        print(f'{step_len=}')

        factor = ((prop_len/step_len - 1) * k) + 1
        rotated_pwm = rotated_pwm * (np.max(rotated_pwm * factor)/pwm_base)

        return rotated_pwm
