import time
import numpy as np

from typing import Optional
from lib.api.client import QualifiersApiClient as ApiClient
from lib.entities.state import Orientation
from lib.robot.odometer import OdometryState
from lib.robot.base import AbstractRobotController
from lib.robot.sensors_reader import SimpleSensorReader
from lib.maze.util import nearest_cell, cell_equals, get_ground_truth_cell_coordinates, nearest_direction


class FeedbackRobotController(AbstractRobotController):

    def __init__(self, sensors_reader: SimpleSensorReader, api_client: ApiClient, delay: Optional[float],
                 api_move_wait_timeout: float = 5.0, api_command_retries: int = 1):
        super(FeedbackRobotController, self).__init__()
        self._api_client = api_client
        self._delay = delay
        self._odometry = OdometryState(np.array([15, 0]), Orientation.NORTH)
        self._sensors_reader = sensors_reader
        self._api_move_timeout = api_move_wait_timeout
        self._api_command_retries = api_command_retries

    @property
    def odometry(self) -> OdometryState:
        """Current cell position and orientation of the robot.
        """
        return self._odometry

    def _is_timeout(self, set_time: float):
        return time.monotonic() - set_time > self._api_move_timeout

    def _wait_linear_move(self, expected: OdometryState):
        cell = self._odometry.cell
        set_time = time.monotonic()
        while (not cell_equals(expected.cell, cell)):
            sensors = self._api_client.read_sensors()
            cell = nearest_cell(
                sensors=sensors, initial_guess=expected.cell, distance_threshold=3.0)
            self._do_delay()
            if (self._is_timeout(set_time)):
                print("Linear move timeout - maybe retry command?")
                return False

        self._odometry = expected
        return True

    def _wait_rotation(self, expected: OdometryState):
        orientation = self._odometry.orientation
        set_time = time.monotonic()
        while (expected.orientation != orientation):
            sensors = self._api_client.read_sensors()
            orientation = nearest_direction(
                sensors=sensors, initial_guess=orientation, threshold=0.5)
            self._do_delay()
            if (self._is_timeout(set_time)):
                print("Rotation move timeout - maybe retry command?")
                return False

        self._odometry = expected
        return True

    def move_forward(self, n: int = 1):
        """Moves robot n cells forward (in local frame).
        """
        for i in range(n):
            for j in range(self._api_command_retries):
                self._api_client.forward_cell()
                expected = self._odometry.move_forward()
                if (self._wait_linear_move(expected=expected)):
                    break

    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        for i in range(n):
            for j in range(self._api_command_retries):
                self._api_client.backward_cell()
                expected = self._odometry.move_backward()
                if (self._wait_linear_move(expected=expected)):
                    break

    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        for i in range(n):
            for j in range(self._api_command_retries):
                self._api_client.left_cell()
                expected = self._odometry.rotate_left()
                if (self._wait_rotation(expected=expected)):
                    break

    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        for i in range(n):
            for j in range(self._api_command_retries):
                self._api_client.right_cell()
                expected = self._odometry.rotatate_right()
                if (self._wait_rotation(expected=expected)):
                    break

    def _do_delay(self):
        if self._delay is not None:
            time.sleep(self._delay)
