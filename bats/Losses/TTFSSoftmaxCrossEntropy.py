from typing import Tuple

from ..AbstractLoss import AbstractLoss
import cupy as cp


class TTFSSoftmaxCrossEntropy(AbstractLoss):
    def __init__(self, tau: float):
        self.__tau: cp.float32 = cp.float32(tau)
        self.__exp_kernel = cp.ElementwiseKernel("float32 t",
                                                 "float32 out",
                                                 f"out = __expf(-t / {tau})",
                                                 "sce_exp_kernel")

        self.__cross_entropy_kernel = cp.ElementwiseKernel("float32 labels_exps, float32 sums",
                                                           "float32 out",
                                                           "out = - __logf(labels_exps / sums)",
                                                           "sce_cross_entropy_kernel")

    def predict(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray) -> cp.ndarray:
        return cp.argmin(spikes_per_neuron, axis=1).flatten()

    def compute_loss(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                     labels: cp.ndarray) -> cp.ndarray:
        exps = self.__exp_kernel(spikes_per_neuron[..., 0])
        sums = cp.sum(exps, axis=1)
        labels_exps = exps[cp.arange(labels.size), labels]
        return self.__cross_entropy_kernel(labels_exps, sums)

    def compute_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                       labels: cp.ndarray) -> cp.ndarray:
        exps = self.__exp_kernel(spikes_per_neuron[..., 0])
        sums = cp.sum(exps, axis=1)
        neg_softmax = -exps / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neg_softmax /= self.__tau
        errors = cp.zeros(spikes_per_neuron.shape, dtype=cp.float32)
        errors[..., 0] = cp.nan_to_num(neg_softmax)
        return errors

    def compute_loss_and_errors(self, spikes_per_neuron: cp.ndarray, n_spike_per_neuron: cp.ndarray,
                                labels: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        exps = self.__exp_kernel(spikes_per_neuron[..., 0])
        sums = cp.sum(exps, axis=1)

        # Loss
        labels_exps = exps[cp.arange(labels.size), labels]
        loss = self.__cross_entropy_kernel(labels_exps, sums)

        # Error
        neg_softmax = -exps / sums[:, cp.newaxis]
        neg_softmax[cp.arange(labels.size), labels] += 1
        neg_softmax /= self.__tau
        errors = cp.zeros(spikes_per_neuron.shape, dtype=cp.float32)
        errors[..., 0] = cp.nan_to_num(neg_softmax)
        return loss, errors
