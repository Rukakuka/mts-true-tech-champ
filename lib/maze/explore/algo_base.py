import enum
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class RefinementPolicy(enum.Enum):
    NO = "no"
    EXACT = "exact"
    APPROX = "approx"


class AbstractMazeExplorer(ABC):
    """Base class for implementing maze exploration (mappping algorithms).
    """
    
    @abstractmethod
    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the exploration (mapping) process.

        Returns:
            np.ndarray: built map of the maze
        """
        pass
