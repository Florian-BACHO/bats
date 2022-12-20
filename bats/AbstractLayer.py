from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import cupy as cp

import numpy as np


WEIGHTS_FILE_SUFFIX = "_weights.npy"

class AbstractLayer(ABC):
    def __init__(self, n_neurons: int, name: str = ""):
        self._n_neurons: int = n_neurons
        self._name: str = name

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        pass

    @property
    @abstractmethod
    def weights(self) -> Optional[cp.ndarray]:
        pass

    @weights.setter
    @abstractmethod
    def weights(self, weights: np.ndarray) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def forward(self, max_simulation: float, training: bool = False) -> None:
        pass

    @abstractmethod
    def backward(self, errors: cp.array) \
            -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        pass

    @abstractmethod
    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass

    def store(self, dir_path: Path) -> None:
        weights = self.weights
        if weights is not None:
            filename = dir_path / (self._name + WEIGHTS_FILE_SUFFIX)
            np.save(filename, weights.get())

    def restore(self, dir_path: Path) -> None:
        filename = dir_path / (self._name + WEIGHTS_FILE_SUFFIX)
        if filename.exists():
            self.weights = np.load(filename)