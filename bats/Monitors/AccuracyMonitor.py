from bats.AbstractMonitor import AbstractMonitor
import numpy as np

class AccuracyMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__("Accuracy (%)", **kwargs)
        self._hits = 0
        self._n_samples = 0

    def add(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        self._hits += np.sum(predictions == targets)
        self._n_samples += targets.shape[0]

    def record(self, epoch) -> float:
        accuracy = self._hits / self._n_samples * 100
        super()._record(epoch, accuracy)
        self._hits = 0
        self._n_samples = 0
        return accuracy
