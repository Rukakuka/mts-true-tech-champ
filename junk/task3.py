import numpy as np
import requests
import enum
from  time import time, sleep
from pynput import keyboard
from dataclasses import dataclass

API_TOKEN = 'b04b0324-04cd-42b6-96af-4326181a29b600009497-e25b-4b6e-a83f-07038b89f61d'

def get_maze():
    maze_list = [[[1253.94, -1254.48],
  [1254.0, -1087.75],
  [1254.0, -921.07],
  [1254.0, -754.4],
  [1254.0, -587.74],
  [1254.0, -421.08],
  [1253.25, -254.97],
  [1253.3, -88.23],
  [1253.41, 78.58],
  [1254.69, 246.06],
  [1254.69, 412.71],
  [1254.91, 578.23],
  [1254.96, 744.97],
  [1254.96, 911.65],
  [1254.89, 1078.41],
  [1254.93, 1245.14]],
 [[1087.28, -1254.48],
  [1086.93, -1088.06],
  [1087.05, -921.32],
  [1087.09, -754.52],
  [1087.19, -587.75],
  [1087.26, -421.02],
  [1086.45, -254.9],
  [1086.56, -88.16],
  [1086.6, 78.64],
  [1087.88, 246.12],
  [1087.95, 412.92],
  [1088.23, 578.23],
  [1088.16, 744.97],
  [1088.23, 911.7],
  [1088.08, 1078.48],
  [1088.2, 1245.21]],
 [[920.6, -1254.48],
  [920.2, -1087.99],
  [920.25, -921.26],
  [920.36, -754.45],
  [920.39, -587.72],
  [921.03, -420.83],
  [921.03, -254.18],
  [921.07, -87.38],
  [921.17, 79.36],
  [921.21, 246.16],
  [921.21, 412.97],
  [921.49, 578.28],
  [921.55, 745.08],
  [921.46, 911.8],
  [921.46, 1078.48],
  [921.52, 1245.21]],
 [[753.87, -1254.44],
  [753.93, -1087.63],
  [754.04, -921.07],
  [754.1, -754.27],
  [753.74, -587.72],
  [754.22, -420.77],
  [752.83, -254.2],
  [754.34, -87.31],
  [754.37, 79.43],
  [754.48, 246.24],
  [754.53, 412.97],
  [752.47, 578.88],
  [754.87, 745.08],
  [754.66, 911.86],
  [754.72, 1078.66],
  [754.61, 1245.6]],
 [[587.11, -1254.5],
  [587.16, -1087.77],
  [587.24, -921.0],
  [587.43, -754.27],
  [587.08, -587.72],
  [587.58, -420.77],
  [587.33, -253.58],
  [587.38, -86.84],
  [587.72, 79.43],
  [587.49, 246.62],
  [587.54, 413.35],
  [587.54, 580.03],
  [588.14, 745.13],
  [588.19, 911.93],
  [587.99, 1078.71],
  [588.03, 1245.51]],
 [[420.44, -1254.5],
  [420.43, -1087.7],
  [420.59, -921.0],
  [420.77, -754.27],
  [420.42, -587.72],
  [420.92, -420.77],
  [420.67, -253.58],
  [420.65, -86.77],
  [420.69, 80.03],
  [420.69, 246.68],
  [420.73, 413.34],
  [420.8, 580.08],
  [418.91, 745.38],
  [421.51, 911.93],
  [421.23, 1078.82],
  [421.3, 1245.56]],
 [[253.76, -1254.5],
  [253.88, -1087.74],
  [253.93, -921.0],
  [254.08, -754.27],
  [254.2, -587.5],
  [254.25, -420.77],
  [254.02, -253.58],
  [253.93, -86.8],
  [253.93, 79.85],
  [253.99, 246.65],
  [253.93, 413.39],
  [254.14, 580.13],
  [254.19, 746.86],
  [254.78, 911.98],
  [254.78, 1078.63],
  [254.84, 1245.43]],
 [[87.03, -1254.43],
  [87.07, -1087.7],
  [86.86, -920.89],
  [87.35, -754.19],
  [87.39, -587.46],
  [87.43, -420.63],
  [87.47, -253.9],
  [87.47, -87.22],
  [87.47, 79.45],
  [87.31, 246.65],
  [87.28, 413.39],
  [87.34, 580.2],
  [87.46, 746.93],
  [85.42, 911.86],
  [85.2, 1079.09],
  [88.17, 1245.43]],
 [[-79.58, -1254.6],
  [-79.51, -1087.79],
  [-79.88, -920.85],
  [-79.81, -754.05],
  [-79.43, -587.32],
  [-79.38, -420.59],
  [-79.12, -253.86],
  [-79.33, -87.23],
  [-79.26, 79.5],
  [-79.42, 246.7],
  [-79.36, 413.5],
  [-79.28, 580.2],
  [-79.23, 746.93],
  [-81.22, 911.86],
  [-81.16, 1078.66],
  [-78.49, 1245.43]],
 [[-246.32, -1254.48],
  [-246.25, -1087.75],
  [-246.25, -921.05],
  [-246.21, -754.32],
  [-246.1, -587.32],
  [-245.85, -420.47],
  [-245.85, -253.81],
  [-245.85, -87.16],
  [-245.39, 79.68],
  [-245.34, 246.42],
  [-245.34, 413.1],
  [-246.08, 580.24],
  [-247.96, 745.3],
  [-247.96, 911.97],
  [-247.89, 1078.71],
  [-245.16, 1245.43]],
 [[-413.12, -1254.44],
  [-413.06, -1087.64],
  [-413.06, -920.99],
  [-412.94, -754.25],
  [-412.76, -587.32],
  [-412.26, -420.34],
  [-412.19, -253.61],
  [-412.19, -86.93],
  [-412.19, 79.74],
  [-412.15, 246.42],
  [-412.08, 413.15],
  [-412.73, 580.24],
  [-414.77, 745.35],
  [-414.7, 912.16],
  [-414.7, 1078.8],
  [-411.82, 1245.43]],
 [[-579.21, -1254.1],
  [-579.15, -1087.36],
  [-579.67, -920.98],
  [-579.62, -754.25],
  [-579.43, -587.32],
  [-578.99, -420.3],
  [-578.96, -253.5],
  [-579.0, -86.96],
  [-579.0, 79.72],
  [-578.95, 246.46],
  [-579.06, 413.23],
  [-579.39, 580.24],
  [-579.19, 746.98],
  [-579.15, 913.71],
  [-581.43, 1078.85],
  [-578.48, 1245.43]],
 [[-745.87, -1254.1],
  [-745.89, -1087.29],
  [-746.48, -920.94],
  [-746.41, -754.14],
  [-746.09, -587.32],
  [-745.75, -420.75],
  [-745.69, -253.45],
  [-745.69, -86.8],
  [-745.65, 80.0],
  [-745.86, 246.54],
  [-745.79, 413.28],
  [-746.05, 580.24],
  [-746.0, 747.04],
  [-745.88, 913.78],
  [-748.11, 1078.85],
  [-745.13, 1245.43]],
 [[-912.54, -1254.1],
  [-912.57, -1087.29],
  [-912.51, -920.49],
  [-912.51, -753.84],
  [-912.75, -587.32],
  [-912.39, -420.75],
  [-912.19, -254.01],
  [-912.15, -87.28],
  [-912.38, 80.05],
  [-912.38, 246.7],
  [-912.52, 413.48],
  [-912.45, 580.21],
  [-912.45, 746.89],
  [-912.56, 913.78],
  [-912.0, 1078.74],
  [-911.93, 1245.47]],
 [[-1079.2, -1254.1],
  [-1079.43, -1087.29],
  [-1079.31, -920.53],
  [-1079.24, -753.79],
  [-1079.4, -587.32],
  [-1079.05, -420.75],
  [-1079.0, -253.95],
  [-1078.88, -87.21],
  [-1078.95, 79.35],
  [-1078.89, 246.15],
  [-1079.25, 413.52],
  [-1079.25, 580.17],
  [-1079.21, 746.98],
  [-1079.23, 913.78],
  [-1078.73, 1078.78],
  [-1078.67, 1245.59]],
 [[-1245.87, -1254.1],
  [-1246.16, -1087.22],
  [-1246.12, -920.49],
  [-1246.2, -753.91],
  [-1246.2, -587.26],
  [-1245.63, -420.62],
  [-1245.63, -253.94],
  [-1245.57, -87.21],
  [-1245.69, 79.57],
  [-1245.69, 246.21],
  [-1245.69, 412.87],
  [-1245.69, 579.54],
  [-1245.95, 747.04],
  [-1245.89, 913.78],
  [-1245.48, 1079.0],
  [-1245.48, 1245.65]]]
    return np.array(maze_list)

class Orientation(enum.IntEnum):
    FORWARD = 0
    RIGHT = 1
    BACKWARD = 2
    LEFT = 3

@dataclass
class SensorsReading:
    front_distance: float
    right_side_distance: float
    left_side_distance: float
    back_distance: float
    left_45_distance: float
    right_45_distance: float
    rotation_pitch: float
    rotation_yaw: float
    rotation_roll: float
    down_x_offset: float
    down_y_offset: float


class Robot:
    _BASE_URL = "http://localhost:8801/api/v1/robot-motors"
    def __init__(self, lenght=1, radius=1, pwm_max = 255, pwm_time=0.11, delay=0.11):
        self.lenght = lenght
        self.radius = radius
        self.pwm_max = pwm_max
        self.pwm_time = pwm_time
        self._delay = delay
        self.current_state = np.array((0, 0, 0))
        self.update_state()

    def move(self, vel):
        # (V, w)
        vel = np.array(vel)
        pwm = self._vel2pwm(vel)
        param = {'l': pwm[1],
                 'l_time': self.pwm_time,
                 'r': pwm[0],
                 'r_time': self.pwm_time}
        self._request('post', 'move', param)

    def read_sensors(self):
        data = self._request('get', 'sensor-data', {'token': API_TOKEN})
        return SensorsReading(**data)
    
    def update_state(self):
        data = self.read_sensors()
        state = np.array((data.down_x_offset, data.down_y_offset, data.rotation_yaw))
        self.current_state = state
 
    
    def _vel2pwm(self, vel):
        V, w = vel
        coeff = np.array((-1,1))
        pwm = (V + coeff * w * self.lenght) / (2 * self.radius)
        pwm = np.clip(pwm, -self.pwm_max, self.pwm_max)
        # (r,l)
        pwm = pwm.astype(int)
        print(f'{pwm=}, {V=}, {w=}')
        return pwm
    
    def _request(self, method ,func, param=None) -> bool:
        url = f"{self._BASE_URL}/{func}"
        if param is not None:
            url = url + "?"
            for key, value in param.items():
                url += f"{key}={value}&"
        # print(url)
        if method == 'post':
            response = requests.post(url)
            sleep(self._delay)
        if method == 'get':
            response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Received status code {response.status_code}")
        return response.json()
    

class PD:
    def __init__(self, kp, kd, max_limit=None, min_limit=None):
        self.kp = kp
        self.kd = kd
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.last_time = None
        self.previous_error = None
        
    def __max_clip(self, value):
            return np.clip(value, -self.max_limit, self.max_limit)

    def __call__(self, error):
        dt = 0 if self.last_time is None else (time() - self.last_time)
        self.last_time = time()
        derivative = 0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error

        output = (self.kp * error) + (self.kd * derivative)
        
        if self.max_limit is not None:
            output = self.__max_clip(output)

        if self.min_limit is not None:
            output = np.sign(output + 0.00001) * max(np.abs(output), self.min_limit)

        return output
    

class CarController(Robot):
    def __init__(self, kp_linear=1, kd_linear=1, kp_angular=1, kd_angular=1, distance_offset=20.0, angle_offset = 0.2, success_rate = 1,
                 **kwargs):
        # print(kwargs)
        super().__init__(**kwargs)
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.distance_offset = distance_offset
        self.angle_offset = angle_offset
        self.maze = get_maze()
        self.position = np.array((15,0))
        self._current_orientation = Orientation.FORWARD
        self.orientation_dict = {0: Orientation.FORWARD, 90: Orientation.RIGHT, 180: Orientation.BACKWARD, -90: Orientation.LEFT}
        self.success_rate = success_rate

    def forward(self, shift=1):
        self._shift('forward', shift)
    def left(self, shift=1):
        self._shift('left', shift)
    def right(self, shift=1):
        self._shift('right', shift)
    def backward(self, shift=1):
        self._shift('backward', shift)

    def seq(self, sequence):
        for seq in sequence:
            match seq:
                case 'f':
                    self.forward()
                case 'b':
                    self.backward()
                case 'l':   
                    self.left()
                case 'r':
                    self.right()

    @property
    def current_cell(self):
        return self.position
    
    @property
    def current_orientation(self) -> Orientation:
        return self._current_orientation

    def _shift_move(self, shift):
        self.update_state()
        current_angle = self.current_state[2]

        index, inverse = self._nearest_direction(current_angle)

        self.position[index] += inverse * shift if index else -inverse * shift
        print(f'{index=}, {inverse=}')
        target_position = self.maze[self.position[0], self.position[1], index]

        if shift > 0:
            pd = PD(self.kp_linear, self.kd_linear, max_limit=300, min_limit=50)
        else:
             pd = PD(self.kp_linear, self.kd_linear*1.8, max_limit=300, min_limit=50)
        error = target_position - self.current_state[index]

        # print(self.current_state[index])
        # print(f"start_move, {index=}, {current_angle=}, {np.abs(current_angle)%360}, {error=}")
        print("start_move")
        success_counter = 0
        while success_counter < self.success_rate[1]:
            self.update_state()
            error = target_position - self.current_state[index]
            speed = pd(error)
            # print(target_position, self.current_state[index], speed)
            self.move((inverse * speed, 0))
            if abs(error) < self.distance_offset:
                success_counter += 1
                print(f'{success_counter=}')

        self.move((0, 0))
        print(f"end_move, {error=}, {self.distance_offset=}")

    def _shift_rotate(self, shift):
        shift = self._nearest_angle(shift * 90)
        print(f'{shift=}')
        self.update_state()
        initial_angle = self.current_state[2]
        target_angle = self._nearest_angle(initial_angle + shift)
        self._current_orientation = self.orientation_dict[target_angle]

        pd = PD(self.kp_angular, self.kd_angular, min_limit=55, max_limit=500)
        error = self._angle_norm(target_angle - self.current_state[2])
        print("start_rotation")

        success_counter = 0
        while success_counter < self.success_rate[0]:
            self.update_state()
            error = self._angle_norm(target_angle - self.current_state[2])
            speed = pd(error)
            # print(self.current_state[2], target_angle, speed)
            self.move((0, speed))

            if abs(error) < self.angle_offset:
                success_counter += 1
                print(f'{success_counter=}')

        self.move((0, 0))
        print(f"end_rotation, {error=}, {self.angle_offset=}")
        
    
    def _shift(self, direction, _shift):
        start_time = time()
        match direction:
            case 'forward':
                self._shift_move(_shift)
                pass
            case 'left':
                self._shift_rotate(-_shift)
                pass
            case 'right':
                self._shift_rotate(_shift)
                pass
            case 'backward':
                self._shift_move(-_shift)
                pass
            case _:
                raise ValueError("Invalid direction")
        sleep(0.1)
        print(f'time: {time() - start_time}s')

    def _angle_norm(self, angle):
        if angle > 180:
            return angle - 360
        elif angle < -180:
            return angle + 360
        return angle
    
    def _nearest_angle(self, angle):
        target_angles = [0, 90, 180, -90]
        
        def angle_distance(a, b):
            return min((a - b) % 360, (b - a) % 360)
        nearest = min(target_angles, key=lambda x: angle_distance(angle, x))
        
        return nearest
    
    def _nearest_direction(self, angle):
        nearest_angle = self._nearest_angle(angle)
        print(f'{nearest_angle=}')
        if nearest_angle in (90, -90):
            index = 1
        else:
            index = 0
        
        if nearest_angle in (0, 90):
            direction = 1
        else: 
            direction = -1

        return index, direction
            
# controller = CarController(
#                            kp_linear=5,    kd_linear=5,
#                            kp_angular=150, kd_angular=0,
#                            distance_offset=20.0,
#                            angle_offset=0.2,
#                            success_rate=1,
#                            #Robot parameters 
#                            lenght=1, radius=1, pwm_max = 255, pwm_time=0.14, delay=0.11
#                            )

# kp_linear=5,    kd_linear=5,  чуть точнее, н медленнее

# controller = CarController(
#                            kp_linear=6,    kd_linear=5,
#                            kp_angular=165, kd_angular=1,
#                            distance_offset=20.0,
#                            angle_offset=0.2,
#                            #Robot parameters 
#                            lenght=1, radius=1, pwm_max = 255, pwm_time=0.1, delay=0
#                            )


from typing import List, Tuple

def batchify_controls(raw_controls: List[str]) -> List[Tuple[str, int]]:
    batchified_controls = []
    sub_start = 0
    sub_end = sub_start + 1
    while(sub_start != len(raw_controls)-1 and sub_end != len(raw_controls)):
        if raw_controls[sub_start] == raw_controls[sub_end]:
            sub_end += 1
        else:
            batchified_controls.append((raw_controls[sub_start], sub_end - sub_start))
            sub_start = sub_end
            sub_end = sub_start + 1
            
    batchified_controls.append((raw_controls[sub_end-1], sub_end - sub_start))
    return batchified_controls



if __name__ == '__main__':
    # def on_press(key):
    #     try:
    #         if key == keyboard.Key.up:
    #             controller.forward(1)
    #         elif key == keyboard.Key.down:
    #             controller.backward(1)
    #         elif key == keyboard.Key.right:
    #             controller.right(1)
    #         elif key == keyboard.Key.left:
    #             controller.left(1)
    #     except AttributeError:
    #         pass

    controller = CarController(
                           kp_linear=15,    kd_linear=5.3,
                           kp_angular=70, kd_angular=6,
                           distance_offset=5.0,
                           angle_offset=0.2,
                           success_rate=(2,3),
                           #Robot parameters 
                           lenght=1, radius=1, pwm_max = 255, pwm_time=0.09, delay=0
                           )

    # with keyboard.Listener(on_press=on_press) as listener:
    #     listener.join()

    sample_input = 'frflffflfrff'
    control = batchify_controls(sample_input)

    for command, k in control:
        match command:
            case 'f':
                controller.forward(k)
            case 'b':
                controller.backward(k)
            case 'l':   
                controller.left(k)
            case 'r':
                controller.right(k)




