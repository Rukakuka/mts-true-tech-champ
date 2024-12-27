import requests
import numpy as np
import time

from lib.entities.sensors import SensorsReading, SemifinalSensorsReading


from typing import Optional


class BaseClient:
    """Basic class to make requests for the API.
    """

    def _request(self, request_type: str, url: str, data: dict = None) -> Optional[requests.Response]:
        """
            Makes a request to the API.
            :param request_type: The type of request to make.
            :param url: The URL to make the request to.
            :param data: The data to send with the request.
            :param max_retries: The maximum number of retries to make.
            :param timeout: The timeout for the request.
            :return: The response from the API.
        """
        for attempt in range(self._max_retries):
            try:
                if request_type == "get":
                    response = requests.get(url, timeout=self._timeout)
                elif request_type == "post":
                    response = requests.post(
                        url, timeout=self._timeout, json=data)
                elif request_type == "put":
                    response = requests.put(
                        url, timeout=self._timeout, json=data)
                else:
                    raise ValueError()
                response.raise_for_status()
                if response.status_code != 200:
                    print(
                        f"Received error status code {response.status_code}")
                else:
                    return response
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.HTTPError, requests.exceptions.RequestException) as err:
                print(f"Attempt {attempt + 1} failed: {err}")
                pass

            if attempt < self._max_retries - 1:
                print(f"Retrying ({attempt + 1}/{self._max_retries})")
                time.sleep(self._sleep_time)
            else:
                print("Max retries reached. Exiting.")

        return None


class QualifiersApiClient(BaseClient):
    def __init__(
            self,
            api_token: str,
            max_retries: int,
            timeout: float,
            sleep_time: float
    ) -> None:
        self._api_token = api_token
        self._url_sensors = f"http://127.0.0.1:8801/api/v1/robot-cells/sensor-data?token={api_token}"
        self._url_forward = f"http://127.0.0.1:8801/api/v1/robot-cells/forward?token={api_token}"
        self._url_backward = f"http://127.0.0.1:8801/api/v1/robot-cells/backward?token={api_token}"
        self._url_left = f"http://127.0.0.1:8801/api/v1/robot-cells/left?token={api_token}"
        self._url_right = f"http://127.0.0.1:8801/api/v1/robot-cells/right?token={api_token}"

        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep_time = sleep_time
        
        self._yaw_offset = 0.

    def init_yaw(self):
        reading = self.read_sensors()
        yaw = reading.rotation_yaw
        self._yaw_offset = reading

    def read_sensors(self) -> SensorsReading:
        """Reads sensors values from the REST API.
        """
        response = self._request(
            request_type="get",
            url=self._url_sensors
        )
        reading = SensorsReading(**response.json())
        reading.rotation_yaw = reading.rotation_yaw - self._yaw_offset
        return reading

    def forward_cell(self):
        self._movement_request(self._url_forward)

    def backward_cell(self):
        self._movement_request(self._url_backward)

    def left_cell(self):
        self._movement_request(self._url_left)

    def right_cell(self):
        self._movement_request(self._url_right)

    def pwm(self, pwm_l: int, pwm_l_dt: float, pwm_r: int, pwm_r_dt: float):
        url = f"http://127.0.0.1:8801/api/v1/robot-motors/move?l={pwm_l}&l_time={pwm_l_dt}&r={pwm_r}&r_time={pwm_r_dt}"
        self._request(
            request_type="post",
            url=url
        )

    def reset_maze(self):
        url = f"http://127.0.0.1:8801/api/v1/maze/restart?token={self._api_token}"
        self._request(
            request_type="post",
            url=url
        )

    def send_maze(self, maze: np.ndarray) -> int:
        url = f"http://127.0.0.1:8801/api/v1/matrix/send?token={self._api_token}"
        response = self._request(
            request_type="post",
            url=url,
            data=maze.tolist()
        )
        return response.json()["Score"]

    def _movement_request(self, url: str):
        self._request(
            request_type="post",
            url=url
        )


class SemifinalApiClient(BaseClient):
    def __init__(self, id: str, robot_ip: str, max_retries: int = 5, timeout: float = 5.0, sleep_time: float = 0.1) -> None:
        self._id = id
        self._robot_ip = robot_ip
        self._base_url = f"http://{robot_ip}"
        self._url_sensors = f"{self._base_url}/sensor"
        self._url_move = f"{self._base_url}/move"
        self._url_motor = f"{self._base_url}/motor"

        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep_time = sleep_time
        
        self._yaw_offset = 0

    def read_sensors(self) -> Optional[SemifinalSensorsReading]:
        """Reads sensors values from the REST API.
        """
        response = self._request(
            request_type="post",
            url=self._url_sensors,
            data={"id": self._id,
                  "type": "all"
                  }
        )
        return SemifinalSensorsReading(response.json(), self._yaw_offset) if response is not None else None

    def update_yaw_offset(self):
        reading = self.read_sensors()
        yaw = reading.rotation_yaw
        self._yaw_offset = yaw

    def forward(self, mm: int):
        self._movement_request("forward", mm)

    def backward(self, mm: int):
        self._movement_request("backward", mm)

    def left(self, degrees: int):
        return self._rotation_request("left", degrees)

    def right(self, degrees: int):
        return self._rotation_request("right", degrees)

    def pwm(self, pwm_l: int, pwm_l_dt: float, pwm_r: int, pwm_r_dt: float):
        response = self._request(
            request_type="put",
            url=self._url_motor,
            data={"id": self._id,
                  "l": int(np.clip(pwm_l, -255, 255)),
                  "r": int(np.clip(pwm_r, -255, 255)),
                  "l_time": int(np.clip(pwm_l_dt, 0, 10000)),
                  "r_time": int(np.clip(pwm_r_dt, 0, 10000))
                  }
        )
        return response.json() if response is not None else None

    def _rotation_request(self, direction: str, degrees: int):
        response = self._request(
            request_type="put",
            url=self._url_move,
            data={"id": self._id,
                  "direction": direction,
                  "len": int(np.clip(degrees, 0, 360))
                  }
        )

    def _movement_request(self, direction: str, mm: int):
        response = self._request(
            request_type="put",
            url=self._url_move,
            data={"id": self._id,
                  "direction": direction,
                  "len": int(np.clip(mm, 0, 10000))
                  }
        )
