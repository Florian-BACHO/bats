from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import cupy as cp

import numpy as np

from bats import AbstractLayer


class AbstractConvLayer(AbstractLayer):
    def __init__(self, neurons_shape: np.ndarray, name: str = ""):
        n_neurons = neurons_shape[0] * neurons_shape[1] * neurons_shape[2]
        super(AbstractConvLayer, self).__init__(n_neurons, name)
        self._neurons_shape: cp.ndarray = cp.array(neurons_shape, dtype=cp.int32)

    @property
    def neurons_shape(self) -> cp.ndarray:
        return self._neurons_shape