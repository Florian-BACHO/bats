from pathlib import Path
from typing import Optional

from bats.AbstractMonitor import AbstractMonitor
import time

class TimeMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__("Time (s)", **kwargs)
        self._start_time = 0

    def start(self) -> None:
        self._start_time = time.time()

    def record(self, epoch) -> float:
        end_time = time.time()
        elapsed_time = end_time - self._start_time
        super()._record(epoch, elapsed_time)
        return elapsed_time