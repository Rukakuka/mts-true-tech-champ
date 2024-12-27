
import time
from lib.robot.sensors_reader import AbstractSensorsReader, SensorsReading


def wait_api_init(sensors_reader: AbstractSensorsReader, sleep_time: float = 1.0):
    print("Wait initialization...")
    while (True):
        try:
            if (sensors_reader.get_reading().is_zero()):
                print(f"Sensor data is all zeros, continue...")
                time.sleep(sleep_time)
                continue
            else:
                print("Initialization OK!")
                return
        except Exception as e:
            print(f"Cannot get sensor data: {e}, continue...")
            time.sleep(sleep_time)
            continue
