from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class SpikeCountLoss(AbstractLoss):
    def __init__(self):
        self.__loss_kernel = cp.ReductionKernel("float32 out_count, float32 out_target",
                                                "float32 loss",
                                                "(out_target - out_count) * (out_target - out_count)",
                                                "a + b",
                                                "loss = a / 2",
                                                "0",
                                                "loss_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        return cp.argmax(n_spike_per_neuron, axis=1)

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     targets: cp.ndarray) -> cp.ndarray:
        float_n_spike_per_neuron = n_spike_per_neuron.astype(cp.float32)
        return self.__loss_kernel(float_n_spike_per_neuron, targets, axis=1)

    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       targets: cp.ndarray) -> cp.ndarray:
        max_n_spike = spikes_per_neuron.shape[2]
        neurons_errors = targets - n_spike_per_neuron.astype(cp.float32)
        return cp.repeat(neurons_errors[:, :, cp.newaxis], repeats=max_n_spike, axis=2)

    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                targets: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        max_n_spike = spikes_per_neuron.shape[2]
        float_n_spike_per_neuron = n_spike_per_neuron.astype(cp.float32)
        neurons_errors = targets - float_n_spike_per_neuron
        loss = self.__loss_kernel(float_n_spike_per_neuron, targets, axis=1)
        errors = cp.repeat(neurons_errors[:, :, cp.newaxis], repeats=max_n_spike, axis=2)
        return loss, errors