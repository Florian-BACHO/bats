from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from bats.AbstractConvLayer import AbstractConvLayer
from bats.CudaKernels.Wrappers.Backpropagation.compute_weights_gradient_conv import compute_weights_gradient_conv
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *
from bats.CudaKernels.Wrappers.Inference.compute_spike_times_conv import compute_spike_times_conv


class PoolingLayer(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        n_x = prev_x // 2
        n_y = prev_y // 2
        neurons_shape: cp.ndarray = np.array([n_x, n_y, prev_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, **kwargs)

        self.__previous_layer: AbstractConvLayer = previous_layer

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__spike_indices: Optional[cp.ndarray] = None

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        return self.__spike_times_per_neuron, self.__n_spike_per_neuron

    @property
    def weights(self) -> Optional[cp.ndarray]:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__spike_indices = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains
        self.__n_spike_per_neuron, self.__spike_times_per_neuron, self.__spike_indices = \
            aggregate_spikes_conv(pre_n_spike_per_neuron, pre_spike_per_neuron, self.__previous_layer.neurons_shape,
                                  self.neurons_shape)

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Propagate errors
        if self.__previous_layer.trainable:
            batch_size, n_neurons, max_n_spike = errors.shape
            pre_n_neurons = self.__previous_layer.n_neurons
            pre_errors = cp.empty((batch_size * pre_n_neurons * max_n_spike // 4,), dtype=cp.float32)
            cp.put(pre_errors, self.__spike_indices.flatten(), errors)
            pre_errors = pre_errors.reshape((batch_size, pre_n_neurons, max_n_spike // 4))
        else:
            pre_errors = None

        return None, pre_errors

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass
