from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class SpikeTimeLoss(AbstractLoss):
    def __init__(self):
        self.__loss_kernel = cp.ReductionKernel("float32 out_count, float32 out_target",
                                                "float32 loss",
                                                "(out_target - out_count) * (out_target - out_count)",
                                                "a + b",
                                                "loss = a / 2",
                                                "0",
                                                "loss_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        return cp.argmin(spikes_per_neuron[..., 0], axis=1)

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     targets: cp.ndarray) -> cp.ndarray:
        spikes_per_neuron_cpy = cp.copy(spikes_per_neuron)
        spikes_per_neuron_cpy[cp.isinf(spikes_per_neuron)] = cp.float32(1.0)
        targets[cp.isinf(targets)] = cp.float32(1.0)
        targets = cp.array(targets[..., 0], dtype=cp.float32)
        loss = self.__loss_kernel(spikes_per_neuron_cpy[..., 0], targets, axis=1)
        return loss

    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       targets: cp.ndarray) -> cp.ndarray:
        spikes_per_neuron_cpy = cp.copy(spikes_per_neuron)
        spikes_per_neuron_cpy[cp.isinf(spikes_per_neuron)] = cp.float32(1.0)
        targets[cp.isinf(targets)] = cp.float32(1.0)
        targets = cp.array(targets, dtype=cp.float32)
        errors = spikes_per_neuron_cpy - targets
        return errors

    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                targets: cp.ndarray) \
            -> Tuple[cp.ndarray, cp.ndarray]:
        spikes_per_neuron_cpy = cp.copy(spikes_per_neuron)
        spikes_per_neuron_cpy[cp.isinf(spikes_per_neuron)] = cp.float32(1.0)
        targets[cp.isinf(targets)] = cp.float32(1.0)
        targets = cp.array(targets, dtype=cp.float32)
        errors = spikes_per_neuron - targets
        loss = self.__loss_kernel(spikes_per_neuron_cpy[..., 0], targets[..., 0], axis=1)
        return loss, errors
