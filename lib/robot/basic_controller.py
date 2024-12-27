import time
import numpy as np

from typing import Optional
from lib.api.client import SemifinalApiClient
from lib.entities.state import Orientation
from lib.robot.odometer import OdometryState
from lib.robot.base import AbstractRobotController


class BasicRobotController(AbstractRobotController):

    def __init__(self, api_client, delay: Optional[float]):
        super(BasicRobotController, self).__init__()
        self._api_client = api_client
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
        for i in range(n):
            self._api_client.forward_cell()
            self._odometry = self._odometry.move_forward()
            self._do_delay()

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        for i in range(n):
            self._api_client.backward_cell()
            self._odometry = self._odometry.move_backward()
            self._do_delay()

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        for i in range(n):
            self._api_client.left_cell()
            self._odometry = self._odometry.rotate_left()
            self._do_delay()

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        for i in range(n):
            self._api_client.right_cell()
            self._odometry = self._odometry.rotatate_right()
            self._do_delay()

    def _do_delay(self):
        if self._delay is not None:
            time.sleep(self._delay)


class SemifinalBasicRobotController(AbstractRobotController):

    _CELL2MM = 168 # We tried this also 130
    _TURN2DEG = 90

    def __init__(self, api_client: SemifinalApiClient, delay: Optional[float]):
        super(SemifinalBasicRobotController, self).__init__()
        self._api_client = api_client
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
        self._api_client.forward(n * self._CELL2MM)
        self._odometry = self._odometry.move_forward()
        self._do_delay()

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        self._api_client.backward(n * self._CELL2MM)
        self._odometry = self._odometry.move_backward()
        self._do_delay()

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        self._api_client.left(n * self._TURN2DEG)
        self._odometry = self._odometry.rotate_left()
        self._do_delay()

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        self._api_client.right(n * self._TURN2DEG)
        self._odometry = self._odometry.rotatate_right()
        self._do_delay()

    def _do_delay(self):
        if self._delay is not None:
            time.sleep(self._delay)

