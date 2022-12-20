from pathlib import Path
from typing import Optional
import numpy as np

from bats import AbstractLayer
from bats.AbstractMonitor import AbstractMonitor
from bats.Network import Network

import cupy as cp

class SpikeCountMonitor(AbstractMonitor):
    def __init__(self, layer_name: str, **kwargs):
        super().__init__(layer_name + " spike counts", **kwargs)
        self._counts = 0.0
        self._n_samples = 0.0

    def add(self, n_spike_per_neuron) -> None:
        self._counts += cp.sum(n_spike_per_neuron).get()
        self._n_samples += n_spike_per_neuron.shape[0]

    def record(self, epoch) -> float:
        avg_count = self._counts / self._n_samples
        super()._record(epoch, avg_count)
        self._counts = 0.0
        self._n_samples = 0.0
        return avg_count