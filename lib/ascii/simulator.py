from flask import Flask, jsonify, request
import os
import threading
import argparse
import time
import logging
from lib.maze.util import get_ground_truth_cell_coordinates


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

# Лабиринт с полуфинала с циклами
maze = [
[8,  10, 2,  2,  10, 7,  13, 2,  2,  7,  13, 10, 2,  10, 2,  7 ],
[1,  7,  9,  5,  11,  5,  10, 6,  9,  5,  10, 10, 6,  12, 9,  9 ],
[9,  9,  9,  12, 12,  8,  10, 7,  1,  10, 10, 10, 7,  9,  9,  9 ],
[9,  9,  9,  9,  9,  5,  11, 9,  9,  8,  10, 7,  9,  9,  9,  9 ],
[9,  9,  5,  3,  5,  10, 10, 3,  9,  9,  8,  6,  5,  6,  9,  9 ],
[9,  5,  7,  14, 8,  10, 10, 6,  9,  14, 5,  10, 10, 10, 6,  9 ],
[9,  12, 5,  10, 6,  12, 8,  11, 14, 8,  2,  11, 8,  10, 10, 3 ],
[9,  1,  10, 10, 10, 6,  1,  2,  7,  9,  9,  8,  6,  8,  10, 3 ],
[9,  9,  8,  2,  2,  10, 6,  5,  6,  9,  5,  6,  12, 9,  12, 9 ],
[9,  5,  6,  9,  9,  13, 7,  8,  7,  5,  10, 7,  9,  1,  6,  9 ],
[5,  2,  10, 6,  5,  7,  9,  9,  1,  11, 8,  6,  1,  6,  8,  3 ],
[8,  6,  8,  7,  8,  3,  1,  6,  1,  7,  9,  8,  6,  12, 9,  14],
[9,  8,  6,  5,  6,  1,  6,  13, 3,  5,  4,  4,  10, 6,  5,  7 ],
[9,  5,  7,  8,  7,  5,  7,  8,  4,  10, 10, 10, 7,  8,  10, 6 ],
[5,  11, 9,  9,  1,  11, 9,  5,  7,  8,  10, 7,  9,  5,  10, 7 ],
[13, 10, 6,  14, 5,  10, 4,  10, 6,  5,  11, 5,  4,  10, 10, 6 ]
]

# Лабиринт с полуфинала без циклов как старовый в гуи симуляторе
# maze = [
#             [8, 10, 7, 8, 10, 2, 11, 8, 10, 2, 10, 11, 8, 10, 2, 7],
#             [1, 7, 9, 5, 7, 5, 10, 6, 12, 5, 10, 10, 6, 12, 9, 9],
#             [9, 9, 9, 12, 9, 8, 10, 2, 0, 10, 10, 10, 7, 9, 9, 9],
#             [9, 9, 9, 9, 9, 5, 11, 9, 9, 8, 10, 7, 9, 9, 9, 9],
#             [9, 9, 5, 3, 5, 10, 10, 3, 14, 9, 8, 6, 5, 6, 9, 9],
#             [9, 5, 7, 14, 8, 10, 10, 6, 8, 6, 5, 10, 10, 10, 6, 9],
#             [9, 12, 5, 10, 6, 12, 8, 10, 6, 8, 2, 11, 8, 10, 10, 6],
#             [9, 1, 10, 10, 10, 6, 1, 2, 7, 9, 9, 8, 6, 8, 10, 7],
#             [9, 9, 8, 7, 8, 10, 6, 5, 6, 9, 5, 6, 12, 9, 12, 9],
#             [9, 5, 6, 9, 9, 13, 7, 8, 7, 5, 10, 7, 9, 1, 6, 9],
#             [5, 2, 10, 6, 5, 7, 9, 9, 1, 11, 8, 6, 1, 6, 8, 3],
#             [8, 6, 8, 7, 8, 6, 1, 6, 5, 7, 9, 8, 6, 12, 9, 14],
#             [9, 8, 6, 5, 6, 8, 6, 13, 7, 5, 6, 5, 10, 6, 5, 7],
#             [9, 5, 7, 8, 7, 5, 7, 8, 4, 10, 10, 10, 7, 8, 10, 6],
#             [5, 11, 9, 9, 1, 11, 9, 5, 7, 8, 10, 7, 9, 5, 10, 7],
#             [13, 10, 6, 14, 5, 10, 4, 10, 6, 5, 11, 5, 4, 10, 10, 6]
#         ]

x = 0
y = 15
yaw = 0


@app.route("/api/v1/robot-cells/right", methods=['POST'])
def api_right():
    global x, y, yaw
    yaw += 90
    if yaw > 180:
        yaw -= 360
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/left", methods=['POST'])
def api_left():
    global x, y, yaw
    yaw -= 90
    if yaw < -180:
        yaw += 360
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/forward", methods=['POST'])
def api_forward():
    global x, y, yaw
    if yaw > -45 and yaw < 45:  # up
        if maze[y][x] in [0, 1, 3, 4, 5, 6, 9, 14]:
            y -= 1
    elif yaw > 45 and yaw < 135:  # right
        if maze[y][x] in [0, 1, 2, 4, 5, 8, 10, 13]:
            x += 1
    elif yaw > 135 or yaw < -135:  # down
        if maze[y][x] in [0, 1, 2, 3, 7, 8, 9, 12]:
            y += 1
    elif yaw > -135 and yaw < -45:  # left
        if maze[y][x] in [0, 2, 3, 4, 6, 7, 10, 11]:
            x -= 1
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/backward", methods=['POST'])
def api_backward():
    global x, y, yaw
    if yaw > -45 and yaw < 45:  # up
        if maze[y][x] in [0, 1, 2, 3, 7, 8, 9, 12]:
            y += 1
    elif yaw > 45 and yaw < 135:  # right
        if maze[y][x] in [0, 2, 3, 4, 6, 7, 10, 11]:
            x -= 1
    elif yaw > 135 or yaw < -135:  # down
        if maze[y][x] in [0, 1, 3, 4, 5, 6, 9, 14]:
            y -= 1
    elif yaw > -135 and yaw < -45:  # left
        if maze[y][x] in [0, 1, 2, 4, 5, 8, 10, 13]:
            x += 1
    return jsonify({"Feedback": True})


@app.route("/move", methods=['PUT'])
def semifinal_api_move():
    global x, y, yaw
    data = request.get_json()

    if data['direction'] == 'forward':
        api_forward()
    elif data['direction'] == 'backward':
        api_backward()
    elif data['direction'] == 'left':
        api_left()
    elif data['direction'] == 'right':
        api_right()
    else:
        pass
    return jsonify({"Feedback": True})


@app.route("/api/v1/matrix/send", methods=['POST'])
def api_matrix():
    global maze
    received_maze = request.get_json()
    score = 0
    for y in range(16):
        for x in range(16):
            if received_maze[y][x] == maze[y][x]:
                score += 1
    return jsonify({"Score": score})


@app.route("/api/v1/maze/restart", methods=['POST'])
def api_restart():
    global x, y, yaw
    x = 0
    y = 15
    yaw = 0
    return jsonify({"Feedback": True})


def get_laser_data(wall, no_wall):
    global x, y, yaw
    no_wall_up = maze[y][x] in [0, 1, 3, 4, 5, 6, 9, 14]
    no_wall_right = maze[y][x] in [0, 1, 2, 4, 5, 8, 10, 13]
    no_wall_down = maze[y][x] in [0, 1, 2, 3, 7, 8, 9, 12]
    no_wall_left = maze[y][x] in [0, 2, 3, 4, 6, 7, 10, 11]

    front = wall
    right = wall
    back = wall
    left = wall

    if yaw > -45 and yaw < 45:  # up
        if no_wall_up:
            front = no_wall
        if no_wall_right:
            right = no_wall
        if no_wall_down:
            back = no_wall
        if no_wall_left:
            left = no_wall
    elif yaw > 45 and yaw < 135:  # right
        if no_wall_up:
            left = no_wall
        if no_wall_right:
            front = no_wall
        if no_wall_down:
            right = no_wall
        if no_wall_left:
            back = no_wall
    elif yaw > 135 or yaw < -135:  # down
        if no_wall_up:
            back = no_wall
        if no_wall_right:
            left = no_wall
        if no_wall_down:
            front = no_wall
        if no_wall_left:
            right = no_wall
    elif yaw > -135 and yaw < -45:  # left
        if no_wall_up:
            right = no_wall
        if no_wall_right:
            back = no_wall
        if no_wall_down:
            left = no_wall
        if no_wall_left:
            front = no_wall
    return front, back, left, right


@app.route("/sensor", methods=['POST'])
def semifinal_api_sensors():
    front, back, left, right = get_laser_data(80, 200)
    return jsonify(
        {
            "laser": {
                "1": back,
                "2": left,
                "3": 65535,
                "4": front,
                "5": right,
                "6": 65535
            },
            "imu": {
                "pitch": 0,
                "yaw": yaw,
                "roll": 0,
            }
        }
    )


@app.route("/api/v1/robot-cells/sensor-data")
def api_sensors():
    front, back, left, right = get_laser_data(50, 100)

    xy = get_ground_truth_cell_coordinates()

    return jsonify(
        {
            "front_distance": front,
            "right_side_distance": right,
            "left_side_distance": left,
            "back_distance": back,
            "left_45_distance": 100,
            "right_45_distance": 100,
            "rotation_pitch": 0,
            "rotation_yaw": yaw,
            "rotation_roll": 0,
            "down_x_offset": xy[y][x][0],
            "down_y_offset": xy[y][x][1]
        }
    )


wall_ascii = [
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


def render_loop():
    global x, y, yaw, running
    output_lines = [[' '] * 66 for i in range(33)]
    while True:
        for y2 in range(16):
            for x2 in range(16):
                for y1 in range(3):
                    for x1 in range(6):
                        output_lines[y2*2+y1][x2*4 +
                                              x1] = wall_ascii[maze[y2][x2]][y1][x1]

        robot = ''
        if yaw > -45 and yaw < 45:  # up
            robot = '⇧'
        elif yaw > 45 and yaw < 135:  # right
            robot = '⇨'
        elif yaw > 135 or yaw < -135:  # down
            robot = '⇩'
        elif yaw > -135 and yaw < -45:  # left
            robot = '⇦'

        output_lines[y*2+1][x*4+2] = robot

        clear_screen()
        for yy in range(33):
            print("".join(output_lines[yy]))

        time.sleep(1/60)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8801)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Running on {args.ip}:{args.port}")
    time.sleep(1.0)
    threading.Thread(target=render_loop).start()
    app.run(args.ip, args.port)
