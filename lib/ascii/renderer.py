import time
from lib.ascii.terminal import WindowTerminal
import os
RATE = 60

WALLS_ASCII = [
    ["      ", "      ", "      "],
    ["      ", "│     ", "      "],
    ["  ──  ", "      ", "      "],
    ["      ", "     │", "      "],
    ["      ", "      ", "  ──  "],
    ["      ", "│     ", "  ──  "],
    ["      ", "     │", "  ──  "],
    ["  ──  ", "     │", "      "],
    ["  ──  ", "│     ", "      "],
    ["      ", "│    │", "      "],
    ["  ──  ", "      ", "  ──  "],
    ["  ──  ", "     │", "  ──  "],
    ["  ──  ", "│    │", "      "],
    ["  ──  ", "│     ", "  ──  "],
    ["      ", "│    │", "  ──  "],
    ["  ──  ", "│    │", "  ──  "],
]


class AsciiRenderer():
    def __init__(self, maze, x, y, yaw):
        self._terminate = False
        self._lines = [[' '] * 66 for i in range(33)]
        self._window = WindowTerminal.create_window()
        self._window.open()
        self._x = x
        self._y = y
        self._yaw = yaw
        self._maze = maze
        

    def render(self):
        while (not self._terminate):
            for y2 in range(16):
                for x2 in range(16):
                    for y1 in range(3):
                        for x1 in range(6):
                            self._lines[y2*2+y1][x2*4 +
                                                 x1] = WALLS_ASCII[self._maze[y2][x2]][y1][x1]

            robot = ''
            if self._yaw > -45 and self._yaw < 45:  # up
                robot = '⇧'
            elif self._yaw > 45 and self._yaw < 135:  # right
                robot = '⇨'
            elif self._yaw > 135 or self._yaw < -135:  # down
                robot = '⇩'
            elif self._yaw > -135 and self._yaw < -45:  # left
                robot = '⇦'
            self._lines[self._y*2+1][self._x*4+2] = robot
            out = '\n'
            for yy in range(33):
                out += ("".join(self._lines[yy]) + "\n")
            self._window.clear()
            self._window.print(out)
            time.sleep(1/RATE)

        self._window.close()

    def update(self, maze, x, y, yaw):
        self._maze = maze
        self._x = x
        self._y = y
        self._yaw = yaw

    def terminate(self):
        self._terminate = True
