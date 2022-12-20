from pathlib import Path
from typing import Optional
import numpy as np

from bats.AbstractMonitor import AbstractMonitor

class SilentLabelsMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__("Silent labels (%)", **kwargs)
        self._silent_labels = 0
        self._n_samples = 0

    def add(self, n_out_spikes: np.ndarray, targets: np.ndarray) -> None:
        labels_counts = np.take_along_axis(n_out_spikes, targets[:, np.newaxis], axis=1)
        self._silent_labels += np.sum(labels_counts == 0)
        self._n_samples += targets.shape[0]

    def record(self, epoch) -> float:
        silent_ratio = self._silent_labels / self._n_samples * 100
        super()._record(epoch, silent_ratio)
        self._silent_labels = 0
        self._n_samples = 0
        return silent_ratio