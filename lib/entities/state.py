import enum


class Orientation(enum.IntEnum):
    """Reprsentation of the robot orientation in the global frame.
    Clockwise order to simplify rotation operations.
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
