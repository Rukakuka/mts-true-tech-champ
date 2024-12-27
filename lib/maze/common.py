import enum

from lib.entities.state import Orientation


MAZE_SIDE = 16


CELL_TYPES = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 0, 1): 5,
    (0, 0, 1, 1): 6,
    (0, 1, 1, 0): 7,
    (1, 1, 0, 0): 8,
    (1, 0, 1, 0): 9,
    (0, 1, 0, 1): 10,
    (0, 1, 1, 1): 11,
    (1, 1, 1, 0): 12,
    (1, 1, 0, 1): 13,
    (1, 0, 1, 1): 14,
    (1, 1, 1, 1): 15,
}


WALLS_DICT = {
        0: {
            Orientation.WEST: False,
            Orientation.NORTH: False,
            Orientation.EAST: False,
            Orientation.SOUTH: False 
        },
        1: {
            Orientation.WEST: True,
            Orientation.NORTH: False,
            Orientation.EAST: False,
            Orientation.SOUTH: False 
        },
        2: {
            Orientation.WEST: False,
            Orientation.NORTH: True,
            Orientation.EAST: False,
            Orientation.SOUTH: False 
        },
        3: {
            Orientation.WEST: False,
            Orientation.NORTH: False,
            Orientation.EAST: True,
            Orientation.SOUTH: False 
        },
        4: {
            Orientation.WEST: False,
            Orientation.NORTH: False,
            Orientation.EAST: False,
            Orientation.SOUTH: True 
        },
        5: {
            Orientation.WEST: True,
            Orientation.NORTH: False,
            Orientation.EAST: False,
            Orientation.SOUTH: True 
        },
        6: {
            Orientation.WEST: False,
            Orientation.NORTH: False,
            Orientation.EAST: True,
            Orientation.SOUTH: True 
        },
        7: {
            Orientation.WEST: False,
            Orientation.NORTH: True,
            Orientation.EAST: True,
            Orientation.SOUTH: False 
        },
        8: {
            Orientation.WEST: True,
            Orientation.NORTH: True,
            Orientation.EAST: False,
            Orientation.SOUTH: False 
        },
        9: {
            Orientation.WEST: True,
            Orientation.NORTH: False,
            Orientation.EAST: True,
            Orientation.SOUTH: False 
        },
        10: {
            Orientation.WEST: False,
            Orientation.NORTH: True,
            Orientation.EAST: False,
            Orientation.SOUTH: True 
        },
        11: {
            Orientation.WEST: False,
            Orientation.NORTH: True,
            Orientation.EAST: True,
            Orientation.SOUTH: True 
        },
        12: {
            Orientation.WEST: True,
            Orientation.NORTH: True,
            Orientation.EAST: True,
            Orientation.SOUTH: False 
        },
        13: {
            Orientation.WEST: True,
            Orientation.NORTH: True,
            Orientation.EAST: False,
            Orientation.SOUTH: True 
        },
        14: {
            Orientation.WEST: True,
            Orientation.NORTH: False,
            Orientation.EAST: True,
            Orientation.SOUTH: True 
        },
        15: {
            Orientation.WEST: True,
            Orientation.NORTH: True,
            Orientation.EAST: True,
            Orientation.SOUTH: True 
        }
    }


class PossibleDirection(enum.IntEnum):
    """Movement directions between cells (without rotations).
    """
    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
