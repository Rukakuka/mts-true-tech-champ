from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DEGREES_TO_RADIANS = np.pi / 90.0
LASER_TO_METERS = 1  # TODO


@dataclass
class SensorsReading:
    """Dataclass representing sensor values from the REST API.
    """
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

    def as_dict(self) -> dict:
        return {
            'front_distance': self.front_distance,
            'right_side_distance': self.right_side_distance,
            'left_side_distance': self.left_side_distance,
            'back_distance': self.back_distance,
            'left_45_distance': self.left_45_distance,
            'right_45_distance': self.right_45_distance,
            'rotation_pitch': self.rotation_pitch,
            'rotation_yaw': self.rotation_yaw,
            'rotation_roll': self.rotation_roll,
            'down_x_offset': self.down_x_offset,
            'down_y_offset': self.down_y_offset
        }


def wrap_angle(angle):
    # Use modulo to wrap the angle
    wrapped = angle % 360

    # Handle negative angles
    if wrapped < 0:
        wrapped += 360

    return wrapped


@dataclass
class SemifinalSensorsReading(SensorsReading):

    def __init__(self, json, yaw_offset: float):
        self.back_distance = json["laser"]["1"]
        self.left_side_distance = json["laser"]["2"]
        self.right_45_distance = json["laser"]["3"]
        self.front_distance = json["laser"]["4"]
        self.right_side_distance = json["laser"]["5"]
        self.left_45_distance = json["laser"]["6"]
        self.rotation_roll = json["imu"]["roll"]
        self.rotation_pitch = json["imu"]["pitch"]
        self.rotation_yaw = wrap_angle(json["imu"]["yaw"] - yaw_offset)
        self.down_x_offset = 0
        self.down_y_offset = 0

    def is_zero(self) -> bool:
        return not np.any(np.array([self.front_distance,
                                    self.right_side_distance,
                                    self.left_side_distance,
                                    self.back_distance,
                                    self.left_45_distance,
                                    self.right_45_distance,
                                    self.rotation_pitch,
                                    self.rotation_yaw,
                                    self.rotation_roll,
                                    self.down_x_offset,
                                    self.down_y_offset]))

    @staticmethod
    def from_qualifiers_format(reading: SensorsReading, yaw_offset: float = 0) -> SemifinalSensorsReading:
        json = {}
        json["laser"] = {}
        json["imu"] = {}

        json["laser"]["1"] = reading.back_distance
        json["laser"]["2"] = reading.left_side_distance
        json["laser"]["3"] = reading.left_45_distance
        json["laser"]["4"] = reading.front_distance
        json["laser"]["5"] = reading.right_side_distance
        json["laser"]["6"] = reading.right_45_distance

        json["imu"]["yaw"] = reading.rotation_yaw
        json["imu"]["roll"] = reading.rotation_roll
        json["imu"]["pitch"] = reading.rotation_pitch

        return SemifinalSensorsReading(json, yaw_offset)
