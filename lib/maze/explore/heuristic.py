import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple

from lib.maze.common import PossibleDirection


class AbstractJunctionHeuristic(ABC):
    """Base class for implementing heuristics to select directions
    at the junctions. Direction with MINIMUM heuristic value is selected.
    """

    @abstractmethod
    def select(self, directions: List[Tuple[PossibleDirection, np.ndarray]]) -> Tuple[PossibleDirection, np.ndarray]:
        """Selects the direction with the MINIMUM heuristic value.

        Args:
            directions (List[Tuple[PossibleDirection, np.ndarray]]): Possible directions and their corresponding next cells

        Returns:
            Tuple[PossibleDirection, np.ndarray]: Select direction and its next cell
        """
        pass
        # if len(directions) == 1:
        #     return directions[0]
        
        # values = [self.calculate(e) for e in directions]
        # idx = np.argmin(values)
        # return directions[idx]
    

class FirstEntityHeuristic(AbstractJunctionHeuristic):
    """Simple heuristic that selects the first direction in the list of options.
    """

    def select(self, directions: List[Tuple[PossibleDirection, np.ndarray]]) -> Tuple[PossibleDirection, np.ndarray]:
        """Selects the direction with the MINIMUM heuristic value.

        Args:
            directions (List[Tuple[PossibleDirection, np.ndarray]]): Possible directions and their corresponding next cells

        Returns:
            Tuple[PossibleDirection, np.ndarray]: Select direction and its next cell
        """
        return directions[0]


class RandomHeuristic(AbstractJunctionHeuristic):
    """Selects random direction in the options list.
    """

    def select(self, directions: List[Tuple[PossibleDirection, np.ndarray]]) -> Tuple[PossibleDirection, np.ndarray]:
        """Selects the direction with the MINIMUM heuristic value.

        Args:
            directions (List[Tuple[PossibleDirection, np.ndarray]]): Possible directions and their corresponding next cells

        Returns:
            Tuple[PossibleDirection, np.ndarray]: Select direction and its next cell
        """
        if len(directions) == 0:
            return directions[0]
        indicies = np.arange(len(directions))
        idx = np.random.choice(indicies)
        return directions[idx]
    

class ManhattanHeuristic(AbstractJunctionHeuristic):
    """Selects the option with the lowest Manhattan distance to the target point.
    """

    def __init__(self, target_point: np.ndarray):
        """Creates a ManhattanHeuristic instance.

        Args:
            target_point (np.ndarray): Target point to calculate the distance to
        """
        super(ManhattanHeuristic, self).__init__()
        self._target_point = target_point.copy()

    def select(self, directions: List[Tuple[PossibleDirection, np.ndarray]]) -> Tuple[PossibleDirection, np.ndarray]:
        """Selects the direction with the MINIMUM heuristic value.

        Args:
            directions (List[Tuple[PossibleDirection, np.ndarray]]): Possible directions and their corresponding next cells

        Returns:
            Tuple[PossibleDirection, np.ndarray]: Select direction and its next cell
        """
        if len(directions) == 0:
            return directions[0]
        values = [np.linalg.norm(e[1] - self._target_point, ord=1) for e in directions]
        idx = np.argmin(values)
        return directions[idx]
