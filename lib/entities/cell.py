from typing import List, Tuple, Optional


class Cell():
    """Reprsentation of cell
           North 1
    West 2         East 0
           South 3
    """

    def __init__(self, east: bool = False, north: bool = False, west: bool = False, south: bool = False):
        self.North = north
        self.East = east
        self.South = south
        self.West = west

    def walls(self):
        return self.east, self.north, self.west, self.south

    def as_value(self):
        if not self.West and not self.North and not self.East and not self.South:
            return 0
        elif self.West and not self.North and not self.East and not self.South:
            return 1
        elif not self.West and self.North and not self.East and not self.South:
            return 2
        elif not self.West and not self.North and self.East and not self.South:
            return 3
        elif not self.West and not self.North and not self.East and self.South:
            return 4
        elif self.West and not self.North and not self.East and self.South:
            return 5
        elif not self.West and not self.North and self.East and self.South:
            return 6
        elif not self.West and self.North and self.East and not self.South:
            return 7
        elif self.West and self.North and not self.East and not self.South:
            return 8
        elif self.West and not self.North and self.East and not self.South:
            return 9
        elif not self.West and self.North and not self.East and self.South:
            return 10
        elif not self.West and self.North and self.East and self.South:
            return 11
        elif self.West and self.North and self.East and not self.South:
            return 12
        elif self.West and self.North and not self.East and self.South:
            return 13
        elif self.West and not self.North and self.East and self.South:
            return 14
        elif self.West and self.North and self.East and self.South:
            return 15
