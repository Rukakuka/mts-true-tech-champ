import signal
from datetime import datetime


from lib.entities.sensors import SensorsReading, SemifinalSensorsReading


class SimpleLogger():
    def __init__(self):
        now = datetime.now().isoformat()
        self._file = None
        try:
            self._file = open(f'log{str(now)}.log', 'w', encoding="utf-8")
            # self._file.close()
        except Exception as e:
            print(f"SimpleLogger: cant open 'log{str(now)}.log' for write")
            raise OSError

    def __del__(self):
        self.close()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def log_sensor_reading(self, sensor_reading: SemifinalSensorsReading | SensorsReading):
        if sensor_reading is None:
            return
        if self._file is None or self._file.closed:
            print("SimpleLogger: the log file is not open, continue")
            return

        try:
            now = datetime.now().isoformat()
            self._file.writelines(f'[{now}] {str(sensor_reading.as_dict())}\n')
            self._file.flush()
        except ValueError as e:
            print(
                f"SimpleLogger: trying to log to a closed file: {e}, continue")
            pass
