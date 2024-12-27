import multiprocessing
import time
import threading

from copy import copy
from abc import ABC, abstractmethod
from typing import Union

from lib.entities.sensors import SensorsReading, SemifinalSensorsReading
from lib.api.client import QualifiersApiClient, SemifinalApiClient
from lib.robot.simple_logger import SimpleLogger


class AbstractSensorsReader(ABC):
    """Class is responsible for fetching sensors readings from the PAI.
    """

    @abstractmethod
    def get_reading(self) -> Union[SensorsReading, SemifinalSensorsReading]:
        """Get latest reading from the sensor.

        Returns:
            SensorsReading: Fetched reading
        """
        pass

    def start(self):
        """Initializes the internal routines of the instance.
        Useful for async readers. 
        """
        pass

    def stop(self):
        """Stops and cleans up the internal routines of the instance.
        Useful for async readers. 
        """
        pass


class SimpleSensorReader(AbstractSensorsReader):
    """Simple synchronous implementation of the sensors reader.
    """

    def __init__(self, api_client: QualifiersApiClient):
        """Initializes the synchronous implementation of the sensors reader.

        Args:
            api_client (ApiClient): API client instance.
        """
        super(SimpleSensorReader, self).__init__()
        self._api_client = api_client

    def get_reading(self) -> SensorsReading:
        return self._api_client.read_sensors()


class SimpleSensorReaderSemiFinal(AbstractSensorsReader):
    """Simple synchronous implementation of the sensors reader.
    """

    def __init__(self, api_client: SemifinalApiClient):
        """Initializes the synchronous implementation of the sensors reader for semifinal.

        Args:
            api_client (ApiClient): API client instance.
        """
        super(SimpleSensorReaderSemiFinal, self).__init__()
        self._api_client = api_client

    def get_reading(self) -> SemifinalSensorsReading:
        return self._api_client.read_sensors()


class SimpleSensorReaderSemiFinalAsync(AbstractSensorsReader):
    """Semifinal asynchronous implementation of the sensors reader.
    """

    def __init__(self, api_client: SemifinalApiClient, run: bool = False, rate_hz: float = 20, logger: SimpleLogger = None):
        super(SimpleSensorReaderSemiFinalAsync, self).__init__()
        self._api_client = api_client
        self._rate = rate_hz
        self._run = run
        self._sensors = None
        self._logger = logger

        self._manager = multiprocessing.Manager()
        self._run_event = self._manager.Event()
        self._new_measurement = self._manager.Event()
        self._new_measurement.clear()

        if run:
            self._run_event.set()
        else:
            self._run_event.clear()

        self._poll_queue = multiprocessing.Queue(maxsize=1)
        self._poll_process = multiprocessing.Process(
            target=self._poll_worker, args=(self._run_event, self._new_measurement, self._logger,))

        if run:
            self._poll_process.start()

    def start(self):
        if self._run is False:
            self._run = True
            self._run_event.set()
            self._poll_process.start()
            print("SimpleSensorReaderSemiFinalAsync start")

    def stop(self):
        self._run = False
        self._run_event.clear()
        print("SimpleSensorReaderSemiFinalAsync stop")
        self._poll_process.terminate()
        self._poll_process.join()

    def get_reading(self) -> SensorsReading:
        if (self._new_measurement.is_set()):
            self._sensors = self._poll_queue.get(block=True)
            self._new_measurement.clear()
        return copy(self._sensors)

    def _poll_worker(self, run_event: threading.Event, measurement: threading.Event, logger: SimpleLogger = None):
        cnt = 0
        drop = 0
        t = time.monotonic()
        while run_event.is_set():
            try:
                sensors = self._api_client.read_sensors()
            except Exception as e:
                print(f"Error reading sensors: {e}")
                drop += 1
                continue
            try:
                _ = self._poll_queue.get(block=False)
            except Exception:
                pass
            self._poll_queue.put(sensors, block=False)
            measurement.set()

            cnt += 1
            if (cnt % 100 == 0):
                dt = time.monotonic() - t
                print(
                    f'SimpleSensorReaderSemiFinalAsync: actual poll rate is {cnt/dt:.3f} hz, dropped {drop} times')
            if logger is not None:
                logger.log_sensor_reading(sensors)
            # time.sleep(1/self._rate)
