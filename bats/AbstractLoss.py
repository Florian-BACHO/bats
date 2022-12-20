from abc import ABC, abstractmethod
from typing import Tuple

import cupy as cp

class AbstractLoss(ABC):
    @abstractmethod
    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       labels: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                labels: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        pass