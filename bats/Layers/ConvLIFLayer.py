from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from bats.AbstractConvLayer import AbstractConvLayer
from bats.CudaKernels.Wrappers.Backpropagation.compute_weights_gradient_conv import compute_weights_gradient_conv
from bats.CudaKernels.Wrappers.Backpropagation.propagate_errors_to_pre_spikes_conv import \
    propagate_errors_to_pre_spikes_conv
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *
from bats.CudaKernels.Wrappers.Inference.compute_spike_times_conv import compute_spike_times_conv


class ConvLIFLayer(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, filters_shape: np.ndarray, tau_s: float, theta: float,
                 delta_theta: float,
                 weight_initializer: Callable[[int, int, int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        filter_x, filter_y, filter_c = filters_shape
        n_x = prev_x - filter_x + 1
        n_y = prev_y - filter_y + 1
        neurons_shape: cp.ndarray = np.array([n_x, n_y, filter_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, **kwargs)

        self.__filters_shape = cp.array([filter_c, filter_x, filter_y, prev_c], dtype=cp.int32)
        self.__previous_layer: AbstractConvLayer = previous_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((filter_c, filter_x, filter_y, prev_c), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(filter_c, filter_x, filter_y, prev_c)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__a: Optional[cp.ndarray] = None
        self.__x: Optional[cp.ndarray] = None
        self.__post_exp_tau: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        return self.__spike_times_per_neuron, self.__n_spike_per_neuron

    @property
    def weights(self) -> Optional[cp.ndarray]:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.__weights = cp.array(weights, dtype=cp.float32)

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__pre_exp_tau_s = None
        self.__pre_exp_tau = None
        self.__pre_spike_weights = None
        self.__a = None
        self.__x = None
        self.__post_exp_tau = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains

        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, self.n_neurons, self.__max_n_spike)
            self.__n_spike_per_neuron = cp.zeros((batch_size, self.n_neurons), dtype=cp.int32)
            self.__spike_times_per_neuron = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s, new_shape), sorted_indices,
                                                      axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau, new_shape), sorted_indices, axis=1)

            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.weights, self.__c, self.__delta_theta_tau,
                                                           self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                                           self.__previous_layer.neurons_shape, self.neurons_shape,
                                                           self.__filters_shape)

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau)
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c, self.__x,
                                 self.__post_exp_tau, self.__tau)
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                     self.__pre_exp_tau_s, self.__pre_exp_tau,
                                                     self.__previous_layer.neurons_shape,
                                                     self.neurons_shape,
                                                     self.__filters_shape,
                                                     errors)

        # Propagate errors
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron,
                                                             pre_spike_per_neuron,
                                                             self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights,
                                                             errors, self.__tau_s, self.__tau,
                                                             self.__previous_layer.neurons_shape,
                                                             self.neurons_shape,
                                                             self.__filters_shape)
        else:
            pre_errors = None

        return weights_grad, pre_errors

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights += delta_weights
