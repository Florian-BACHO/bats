from pathlib import Path
from typing import Optional
import numpy as np

from bats.AbstractMonitor import AbstractMonitor

class SilentNeuronsMonitor(AbstractMonitor):
    def __init__(self, layer_name: str, **kwargs):
        super().__init__(layer_name + " silents (%)", **kwargs)
        self._total_counts: Optional[np.ndarray] = None

    def add(self, n_spikes: np.ndarray) -> None:
        if self._total_counts is None:
            self._total_counts = np.sum(n_spikes, axis=0)
            return
        self._total_counts += np.sum(n_spikes, axis=0)

    def record(self, epoch) -> float:
        silent_ratio = np.mean(self._total_counts == 0) * 100
        super()._record(epoch, silent_ratio)
        self._total_counts = None
        return silent_ratio