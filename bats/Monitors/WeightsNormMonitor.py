import numpy as np
import cupy as cp

from bats.AbstractMonitor import AbstractMonitor

class WeightsNormMonitor(AbstractMonitor):
    def __init__(self, layer_name: str, **kwargs):
        super().__init__(layer_name + " weights norm", **kwargs)
        self.__norm = None

    def add(self, weights: np.ndarray) -> None:
        self.__norm = cp.linalg.norm(weights).get()

    def record(self, epoch) -> float:
        super()._record(epoch, self.__norm)
        return self.__norm