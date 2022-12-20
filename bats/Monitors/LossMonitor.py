from pathlib import Path
from typing import Optional

from bats.AbstractMonitor import AbstractMonitor
import numpy as np

class LossMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__("Loss", **kwargs)
        self._losses_cumul = 0.0
        self._n_valid_loss = 0

    def add(self, losses: np.ndarray) -> None:
        valid_losses_mask = np.isfinite(losses)
        valid_losses = losses[valid_losses_mask]
        self._losses_cumul += np.sum(valid_losses)
        self._n_valid_loss += np.sum(valid_losses_mask)

    def record(self, epoch) -> float:
        loss = self._losses_cumul / self._n_valid_loss
        super()._record(epoch, loss)
        self._losses_cumul = 0.0
        self._n_valid_loss = 0
        return loss