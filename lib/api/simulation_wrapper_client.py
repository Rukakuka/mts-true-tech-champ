import requests
import numpy as np
import time

from lib.robot.sensors_reader import SimpleSensorReaderSemiFinalAsync
from lib.entities.sensors import SensorsReading, SemifinalSensorsReading
from lib.api.client import BaseClient

from typing import Optional


class SimulationWrapperClient(BaseClient):
    """ This class has semifinal API interface while implementing
    simulator API calls and converting all values

    NOTE: len parameter in forward(), backward(), left() and right() methods
    is not available! (yet)

    """

    def __init__(
            self,
            api_token: str,
            max_retries: int,
            timeout: float,
            sleep_time: float
    ) -> None:
        self._api_token = api_token
        self._url_sensors = f"http://127.0.0.1:8801/api/v1/robot-cells/sensor-data?token={api_token}"
        self._url_forward = f"http://127.0.0.1:8801/api/v1/robot-cells/forward?token={api_token}"
        self._url_backward = f"http://127.0.0.1:8801/api/v1/robot-cells/backward?token={api_token}"
        self._url_left = f"http://127.0.0.1:8801/api/v1/robot-cells/left?token={api_token}"
        self._url_right = f"http://127.0.0.1:8801/api/v1/robot-cells/right?token={api_token}"

        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep_time = sleep_time

        self._started = False
        self._yaw_offset = 0
        self._down_x_offset = None
        self._down_y_offset = None
        self._yaw = None
        self._prev_time = time.monotonic()
        self._speed_left = 0
        self._speed_right = 0
        self._encoder_left = 0
        self._encoder_right = 0

    def update_yaw_offset(self):
        reading = self.read_sensors()
        yaw = reading.rotation_yaw
        self._yaw_offset = yaw

    def _calculate_encoder_values(self, x: float, y: float, angle: float):
        now = time.monotonic()
        dt = now - self._prev_time

        radius = 0.008  # wheel radius in meters
        width = 0.06    # axle length in meters

        # Calculate the change in position and orientation
        dx = (x - self._down_x_offset)/1000.
        dy = -(y - self._down_y_offset)/1000.

        dtheta = (angle - self._yaw)*np.pi/180
        angle *= np.pi/180

        # Calculate linear and angular velocity
        # Linear velocity is the magnitude of position change
        linear_velocity = np.sqrt(dx*dx + dy*dy)

        # If moving backwards, make linear velocity negative
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        forward = rotation_matrix @ np.array([dx, 0])
        left = rotation_matrix @ np.array([0, 1])
        up = np.cross(forward, left)
        if (angle > -np.pi/2 and angle < np.pi/2):
            linear_velocity *= np.sign(up)
        else:
            linear_velocity *= -np.sign(up)

        # Angular velocity is the change in orientation
        angular_velocity = dtheta

        # Calculate wheel speeds using differential drive kinematics
        # V_l = (2V - ωL) / 2R
        # V_r = (2V + ωL) / 2R
        # where V is linear velocity, ω is angular velocity
        # L is axle length, R is wheel radius

        dt = 1 / 2

        speed_left = (2 * linear_velocity - angular_velocity * width)
        speed_right = (2 * linear_velocity + angular_velocity * width)

        self._encoder_left += (self._speed_left +
                               speed_left)/2 * dt / (2 * radius)
        self._encoder_right += (self._speed_right +
                                speed_right)/2 * dt / (2 * radius)

        self._speed_left = speed_left
        self._speed_right = speed_right

    def read_sensors(self) -> SemifinalSensorsReading:
        """Reads sensors values from the REST API.
        """
        response = self._request(
            request_type="get",
            url=self._url_sensors
        )
        reading = SensorsReading(**response.json())
        reading.rotation_yaw = reading.rotation_yaw - self._yaw_offset

        if (self._started is True):
            self._calculate_encoder_values(
                reading.down_x_offset, reading.down_y_offset, reading.rotation_yaw)
        else:
            self._start_x = reading.down_x_offset
            self._start_y = reading.down_y_offset
            self._started = True

        self._prev_time = time.monotonic()
        self._down_x_offset = reading.down_x_offset
        self._down_y_offset = reading.down_y_offset
        self._yaw = reading.rotation_yaw

        # print(f"Left [{self._encoder_left:3.3f}, Right {self._encoder_right:3.3f}]")

        return SemifinalSensorsReading.from_qualifiers_format(reading)

    def get_encoder_values(self):
        return self._encoder_left, self._encoder_right

    def forward(self, len: float = 180):
        self._movement_request(self._url_forward)

    def backward(self, len: float = 180):
        self._movement_request(self._url_backward)

    def left(self, len: float = 90):
        self._movement_request(self._url_left)

    def right(self, len: float = 90):
        self._movement_request(self._url_right)

    def pwm(self, pwm_l: int, pwm_l_dt: float, pwm_r: int, pwm_r_dt: float):
        pwm_l_dt = pwm_l_dt/1000  # ms to s
        pwm_r_dt = pwm_r_dt/1000
        pwm_l = int(np.clip(pwm_l, -255, 255))
        pwm_r = int(np.clip(pwm_r, -255, 255))
        url = f"http://127.0.0.1:8801/api/v1/robot-motors/move?token={self._api_token}&l={pwm_l}&l_time={pwm_l_dt}&r={pwm_r}&r_time={pwm_r_dt}"
        self._request(
            request_type="post",
            url=url
        )

    def _movement_request(self, url: str):
        self._request(
            request_type="post",
            url=url
        )
