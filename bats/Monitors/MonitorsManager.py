from typing import List, Dict
import numpy as np

from bats.AbstractMonitor import AbstractMonitor


class MonitorsManager:
    def __init__(self, monitors: List[AbstractMonitor] = [], print_prefix: str = ""):
        self._monitors: List[AbstractMonitor] = monitors
        self._print_prefix = print_prefix

    def add(self, monitor: AbstractMonitor) -> None:
        self._monitors.append(monitor)

    def print(self, epoch: float, decimal: int = 1) -> None:
        to_print = self._print_prefix + f"Epoch {np.around(epoch, decimal)}"
        for i, monitor in enumerate(self._monitors):
            to_print += (" | " if i == 0 else ", ") + str(monitor)
        print(to_print)

    def record(self, epoch: float) -> Dict:
        return {monitor: monitor.record(epoch) for monitor in self._monitors}

    def export(self) -> None:
        for monitor in self._monitors:
            monitor.export()