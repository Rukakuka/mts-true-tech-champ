from flask import Flask, jsonify, request
from lib.ascii.renderer import AsciiRenderer
import threading
import logging
import time
from lib.maze.util import get_ground_truth_cell_coordinates

DEFAULT_X = 0
DEFAULT_Y = 15
DEFAULT_YAW = 0


class SimulatorApp():
    def __init__(self, maze=[], x=DEFAULT_X, y=DEFAULT_Y, yaw=DEFAULT_YAW, app=None):
        self._log = logging.getLogger('werkzeug')
        self._log.setLevel(logging.ERROR)
        self._app = app
        self._x = x
        self._y = y
        self._yaw = yaw
        self._maze = maze

        self._compare_maze_result = False
        self._score = 0

        self._ascii_renderer_thread = None
        self._app_thread = None
        self._ascii_renderer = None

        self._maze_reset_counter = 0
        self._maze_last_reset_time = time.monotonic()
        self._maze_start_times = []
        self._maze_end_times = []
        self._maze_was_completed = []
        self._robot_started = False
        self._robot_finished = False

    def _run_app(self, stop_event):
        self._request_render_update()
        self._check_robot_position()
        self._app.run("0.0.0.0", 8801)

    def _reset(self, maze, x=DEFAULT_X, y=DEFAULT_Y, yaw=DEFAULT_YAW):
        self._maze_last_reset_time = time.monotonic()
        self._maze_reset_counter += 1
        self._robot_started = False
        self._robot_finished = False
        self._maze = maze
        self._x = x
        self._y = y
        self._yaw = yaw
        self._request_render_update()

    def _check_robot_position(self):
        started = not (self._x == 0 and self._y == 15)
        finished = ((self._x == 7 or self._x == 8) and
                    (self._y == 7 or self._y == 8))

        if (started):
            if (not self._robot_started):
                print(f"Robot started! State: ({self._x}, {self._y})")
                self._maze_start_times += [time.monotonic()]
                self._robot_started = True

        if (finished):
            if (not self._robot_finished):
                print(f"Robot finished! State: ({self._x}, {self._y})")
                self._maze_end_times += [time.monotonic()]
                self._maze_was_completed += [True]
                self._robot_finished = True

    def _compare_mazes(self, expected_maze, received_maze):
        score = 0
        if len(expected_maze) != len(received_maze):
            print(
                f"The mazes have different dimensions {len(expected_maze)=} and {len(received_maze)=}")
            return
        for i, (expected_row, received_row) in enumerate(zip(expected_maze, received_maze)):
            if len(expected_row) != len(received_row):
                print(f"Rows at position {i} have different lengths.")
                break

            for j, (expected_cell, received_cell) in enumerate(zip(expected_row, received_row)):
                if expected_cell != received_cell:
                    print(
                        f"Difference at row {i}, column {j}: expected {expected_cell}, got {received_cell}")
                else:
                    score += 1
        return score

    def _request_render_update(self):
        if self._ascii_renderer is not None:
            self._ascii_renderer.update(
                self._maze, self._x, self._y, self._yaw)

    def run(self, render=False):
        if render:
            self._ascii_renderer = AsciiRenderer(
                self._maze, self._x, self._y, self._yaw)
            self._ascii_renderer_thread = threading.Thread(
                target=self._ascii_renderer.render)
            self._ascii_renderer_thread.start()

        stop_event = threading.Event()
        self._app_thread = threading.Thread(
            target=self._run_app, args=(stop_event,))
        self._app_thread.start()

    def terminate(self):
        if self._app_thread is not None:
            self._app_thread.join()
        if self._ascii_renderer_thread is not None:
            self._ascii_renderer.terminate()
            self._ascii_renderer_thread.join()

    def reset_maze_and_statistics(self, maze, x=DEFAULT_X, y=DEFAULT_Y, yaw=DEFAULT_YAW):
        self._reset(maze, x, y, yaw)
        self._maze_last_reset_time = time.monotonic()
        self._maze_reset_counter = 0
        self._maze_start_times = []
        self._maze_end_times = []
        self._maze_was_completed = []

    def reset(self, maze, x=DEFAULT_X, y=DEFAULT_Y, yaw=DEFAULT_YAW):
        self._reset(maze, x, y, yaw)

    def get_maze_statistics(self):
        if (len(self._maze_start_times) > len(self._maze_end_times)):
            print(f"Seems like robot hasn't succeded: started {len(self._maze_start_times)} times, but finished {len(self._maze_end_times)} times")
            for _ in range(len(self._maze_end_times), len(self._maze_start_times)-len(self._maze_end_times)):
                self._maze_end_times += [time.monotonic()]
                self._maze_was_completed += [False]
        elif (len(self._maze_start_times) < len(self._maze_end_times)):
            raise ValueError(
                "Something is very wrong, robot has finished before even staring!")

        return self._maze_start_times, self._maze_end_times, self._maze_was_completed

    def get_reset_counter(self):
        return self._maze_reset_counter

    def get_score(self):
        return self._score

    def get_compare_maze_result(self):
        return self._compare_maze_result


app = Flask(__name__)
sim = SimulatorApp(app=app)


@app.route("/api/v1/matrix/send", methods=['POST'])
def api_matrix():
    received_maze = request.get_json()
    sim._score = sim._compare_mazes(sim._maze, received_maze)
    sim._compare_maze_result = sim._score == 256
    print(f'Your score: {sim._score}, {sim._compare_maze_result=}')
    return jsonify({"Score": sim._score})


@app.route("/api/v1/maze/restart", methods=['POST'])
def api_restart():
    print(f"Maze reset!")
    sim.reset(sim._maze)
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/sensor-data")
def api_sensors():
    no_wall_up = sim._maze[sim._y][sim._x] in [0, 1, 3, 4, 5, 6, 9, 14]
    no_wall_right = sim._maze[sim._y][sim._x] in [
        0, 1, 2, 4, 5, 8, 10, 13]
    no_wall_down = sim._maze[sim._y][sim._x] in [
        0, 1, 2, 3, 7, 8, 9, 12]
    no_wall_left = sim._maze[sim._y][sim._x] in [
        0, 2, 3, 4, 6, 7, 10, 11]

    front = 50
    right = 50
    back = 50
    left = 50

    if sim._yaw > -45 and sim._yaw < 45:  # up
        if no_wall_up:
            front = 100
        if no_wall_right:
            right = 100
        if no_wall_down:
            back = 100
        if no_wall_left:
            left = 100
    elif sim._yaw > 45 and sim._yaw < 135:  # right
        if no_wall_up:
            left = 100
        if no_wall_right:
            front = 100
        if no_wall_down:
            right = 100
        if no_wall_left:
            back = 100
    elif sim._yaw > 135 or sim._yaw < -135:  # down
        if no_wall_up:
            back = 100
        if no_wall_right:
            left = 100
        if no_wall_down:
            front = 100
        if no_wall_left:
            right = 100
    elif sim._yaw > -135 and sim._yaw < -45:  # left
        if no_wall_up:
            right = 100
        if no_wall_right:
            back = 100
        if no_wall_down:
            left = 100
        if no_wall_left:
            front = 100

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
            "rotation_yaw": sim._yaw,
            "rotation_roll": 0,
            "down_x_offset": xy[sim._y][sim._x][0],
            "down_y_offset": xy[sim._y][sim._x][1]
        }
    )


@app.route("/api/v1/robot-cells/right", methods=['POST'])
def api_right():
    sim._yaw += 90
    if sim._yaw > 180:
        sim._yaw -= 360
    sim._request_render_update()
    sim._check_robot_position()
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/left", methods=['POST'])
def api_left():
    sim._yaw -= 90
    if sim._yaw < -180:
        sim._yaw += 360
    sim._request_render_update()
    sim._check_robot_position()
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/forward", methods=['POST'])
def api_forward():
    if sim._yaw > -45 and sim._yaw < 45:  # up
        if sim._maze[sim._y][sim._x] in [0, 1, 3, 4, 5, 6, 9, 14]:
            sim._y -= 1
    elif sim._yaw > 45 and sim._yaw < 135:  # right
        if sim._maze[sim._y][sim._x] in [0, 1, 2, 4, 5, 8, 10, 13]:
            sim._x += 1
    elif sim._yaw > 135 or sim._yaw < -135:  # down
        if sim._maze[sim._y][sim._x] in [0, 1, 2, 3, 7, 8, 9, 12]:
            sim._y += 1
    elif sim._yaw > -135 and sim._yaw < -45:  # left
        if sim._maze[sim._y][sim._x] in [0, 2, 3, 4, 6, 7, 10, 11]:
            sim._x -= 1
    sim._request_render_update()
    sim._check_robot_position()
    return jsonify({"Feedback": True})


@app.route("/api/v1/robot-cells/backward", methods=['POST'])
def api_backward():
    if sim._yaw > -45 and sim._yaw < 45:  # up
        if sim._maze[sim._y][sim._x] in [0, 1, 2, 3, 7, 8, 9, 12]:
            sim._y += 1
    elif sim._yaw > 45 and sim._yaw < 135:  # right
        if sim._maze[sim._y][sim._x] in [0, 2, 3, 4, 6, 7, 10, 11]:
            sim._x -= 1
    elif sim._yaw > 135 or sim._yaw < -135:  # down
        if sim._maze[sim._y][sim._x] in [0, 1, 3, 4, 5, 6, 9, 14]:
            sim._y -= 1
    elif sim._yaw > -135 and sim._yaw < -45:  # left
        if sim._maze[sim._y][sim._x] in [0, 1, 2, 4, 5, 8, 10, 13]:
            sim._x += 1
    sim._request_render_update()
    sim._check_robot_position()
    return jsonify({"Feedback": True})
