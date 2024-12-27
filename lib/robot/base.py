from abc import ABC, abstractmethod
from lib.robot.odometer import OdometryState


class AbstractRobotController(ABC):
    """Abstract class for defining controllers to move robot from cell to cell using API.
    """
    
    @property
    @abstractmethod
    def odometry(self) -> OdometryState:
        """Current cell position and orientation of the robot.
        """
        pass

    @abstractmethod
    def move_forward(self, n: int = 1):
        """Moves robot n cells forward (in local frame).
        """
        pass

    @abstractmethod
    def move_backward(self, n: int = 1):
        """Moves robot n cells backward (in local frame).
        """
        pass

    @abstractmethod
    def rotate_left(self, n: int = 1):
        """Rotates robot to the left (counter-clockwise) in-place n times.
        """
        pass

    @abstractmethod
    def rotate_right(self, n: int = 1):
        """Rotates robot to the right (clockwise) in-place n times.
        """
        pass
