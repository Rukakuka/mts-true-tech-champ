from __future__ import annotations

import numpy as np

from lib.entities.state import Orientation


class OdometryState:
    """Class that defines localization state,
    cell coordinates and robot orientation,
    and performs immutable operations on them.
    """

    def __init__(self,
                 cell: np.ndarray,
                 orientation: Orientation):
        self._cell = cell
        self._orientation = orientation

    @property
    def cell(self) -> np.ndarray:
        return self._cell.copy()
    
    @property
    def orientation(self) -> Orientation:
        return self._orientation
    
    def rotatate_right(self) -> OdometryState:
        value = self._orientation.value
        value = (value + 1) % 4
        new_orientation = Orientation(value)
        return OdometryState(self._cell.copy(), new_orientation)

    def rotate_left(self) -> OdometryState:
        value = self._orientation.value
        value = value - 1
        if value == -1:
            value = 3
        new_orientation = Orientation(value)
        return OdometryState(self._cell.copy(), new_orientation)

    def move_forward(self) -> OdometryState:
        return OdometryState(self._linear_movement(np.array([-1, 0])),
                             self._orientation)
    
    def move_backward(self) -> OdometryState:
        return OdometryState(self._linear_movement(np.array([1, 0])),
                             self._orientation)

    def _build_rotation_matrix(self) -> np.ndarray:
        orientation  = self._orientation
        angle = 0.
        if orientation == Orientation.NORTH:
            angle = 0.
        elif orientation == Orientation.EAST:
            angle = -np.pi / 2.
        elif orientation == Orientation.SOUTH:
            angle = -np.pi
        elif orientation == Orientation.WEST:
            angle = np.pi / 2.
        else:
            raise RuntimeError(f"Happened something that never could happen")
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return rotation_matrix
    
    def _linear_movement(self, displacement: np.ndarray) -> np.ndarray:
        rotation_matrix = self._build_rotation_matrix()
        current_cell = self._cell
        displacement = rotation_matrix @ displacement
        new_cell = current_cell + displacement.astype(int)
        return new_cell
