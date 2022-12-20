from typing import Tuple

from .SpikeCountLoss import SpikeCountLoss
from ..AbstractLoss import AbstractLoss
import cupy as cp


class SpikeCountClassLoss(SpikeCountLoss):
    def __init__(self, target_true: float, target_false: float):
        super().__init__()

        self.__target_true: cp.float32 = cp.float32(target_true)
        self.__target_false: cp.float32 = cp.float32(target_false)

    def __compute_targets(self, n_spike_per_neuron: cp.ndarray, labels: cp.ndarray):
        targets = cp.full(n_spike_per_neuron.shape, self.__target_false)
        targets[cp.arange(labels.size), labels] = self.__target_true
        return targets

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        targets = self.__compute_targets(n_spike_per_neuron, labels)
        return super().compute_loss(spikes_per_neuron, n_spike_per_neuron, targets)

    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       labels: cp.ndarray) -> cp.ndarray:
        targets = self.__compute_targets(n_spike_per_neuron, labels)
        return super().compute_errors(spikes_per_neuron, n_spike_per_neuron, targets)

    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                labels: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        targets = self.__compute_targets(n_spike_per_neuron, labels)
        return super().compute_loss_and_errors(spikes_per_neuron, n_spike_per_neuron, targets)