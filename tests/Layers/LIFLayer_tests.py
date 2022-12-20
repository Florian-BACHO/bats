import unittest
from pathlib import Path

import numpy as np
import cupy as cp
from brian2 import *

from bats.Layers import InputLayer, LIFLayer
from bats.CudaKernels.Wrappers.Inference import get_sorted_spikes_indices


class LifLayer_tests(unittest.TestCase):
    def compute_spike_times_brian2(self, n_pre, spike_indices, spike_times, n_post, tau_s, tau, theta, delta_theta,
                                   weights, max_simulation):
        dt = 1e-6 * second
        mask = np.isfinite(spike_times)
        input_generator = SpikeGeneratorGroup(n_pre, spike_indices[mask], spike_times[mask] * second, dt=dt)
        layer = NeuronGroup(n_post, '''
                                      du/dt = (-u / tau + v) / second : 1
                                      dv/dt = -v / (tau_s * second) : 1
                                      b: 1
                                      ''',
                            threshold='u >= theta',
                            reset="u -= delta_theta", dt=dt, method="exact")
        synapses = Synapses(input_generator, layer, "w : 1", on_pre="v_post += w",
                            dt=dt)
        synapses.connect()
        synapses.w = weights.T.flatten()
        spike_monitor = SpikeMonitor(layer)
        run(max_simulation * second)
        return spike_monitor.i, spike_monitor.t / second

    def test_forward(self):
        batch_size = 1
        n_pre = 1000
        n_post = 50
        n_spike = 5
        tau_s = np.float32(100e-3)
        tau = np.float32(2 * tau_s)
        theta = 0.1
        delta_theta = 0.2
        max_simulation = 500e-3

        np.random.seed(42)
        n_spike_per_neuron = np.random.randint(0, n_spike, size=(batch_size, n_pre))
        spike_per_neuron = np.full((batch_size, n_pre, n_spike), np.inf, dtype=cp.float32)
        for s_spikes, s_n_spike in zip(spike_per_neuron, n_spike_per_neuron):
            for n_spikes, n in zip(s_spikes, s_n_spike):
                n_spikes[:n] = np.sort(np.random.uniform(low=0.0, high=max_simulation, size=(n,)))

        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(cp.array(spike_per_neuron),
                                                                                    cp.array(n_spike_per_neuron))
        sorted_spike_indices = (sorted_indices.astype(cp.int32) // spike_per_neuron.shape[2])
        sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)

        weights_cpu = np.random.normal(loc=0, scale=1.0, size=(n_post, n_pre))
        weights_gpu = cp.array(weights_cpu, dtype=cp.float32)
        weight_initializer = lambda n_post, n_pre: weights_gpu

        input_layer = InputLayer(n_neurons=n_pre)
        hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=n_post, tau_s=tau_s, theta=theta,
                                delta_theta=delta_theta,
                                weight_initializer=weight_initializer, max_n_spike=128)

        input_layer.set_spike_trains(spike_per_neuron, n_spike_per_neuron)
        input_layer.forward(max_simulation)
        hidden_layer.forward(max_simulation)

        spike_times_per_neuron, n_spike_per_neuron = hidden_layer.spike_trains
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(spike_times_per_neuron,
                                                                                    n_spike_per_neuron)

        spike_indices = sorted_indices // spike_times_per_neuron.shape[-1]
        spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)

        true_spike_indices, true_spike_times = self.compute_spike_times_brian2(n_pre, sorted_spike_indices.get()[0],
                                                                               sorted_spike_times.get()[0], n_post,
                                                                               tau_s, tau, theta, delta_theta,
                                                                               weights_cpu, max_simulation)

        self.assertTrue(len(spike_indices.get()[0]) == len(true_spike_indices))
        self.assertTrue((spike_indices.get()[0] == true_spike_indices).all())
        self.assertTrue(np.allclose(true_spike_times, spike_times.get()[0], atol=1e-2))

    def ttfs_mse(self, spike_time_per_neuron, spike_targets):
        error = cp.zeros(spike_time_per_neuron.shape, dtype=cp.float32)
        first_spikes = spike_time_per_neuron[:, :, 0]
        error[:, :, 0] = (first_spikes - spike_targets)
        return error

    def test_backward(self):
        n_pre = 7
        n_post = 5
        tau_s = 10e-3
        theta = 0.01
        delta_theta = theta
        max_n_spike = 8
        n_epoch = 200
        max_simulation_time = cp.float32(0.02)
        spike_per_neuron = np.array([[[0.0, np.inf],
                                      [3e-3, np.inf],
                                      [6e-3, np.inf],
                                      [8e-3, np.inf],
                                      [9e-3, np.inf],
                                      [11e-3, np.inf],
                                      [17e-3, np.inf]]])
        n_spike_per_neuron = np.array([[1, 1, 1, 1, 1, 1, 1]])
        spike_targets = cp.array([[4e-3, 6e-3, 8e-3, 12e-3, 16e-3]], dtype=cp.float32)
        learning_rate = 3e3

        input_layer = InputLayer(n_neurons=n_pre)
        hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=n_post, tau_s=tau_s, theta=theta,
                                delta_theta=delta_theta,
                                weight_initializer=lambda n_post, n_pre: cp.full((n_post, n_pre), 1.0,
                                                                                 dtype=cp.float32),
                                max_n_spike=max_n_spike)
        input_layer.set_spike_trains(spike_per_neuron, n_spike_per_neuron)

        for epoch in range(n_epoch):
            hidden_layer.reset()
            hidden_layer.forward(max_simulation_time)

            spike_per_neuron, _ = hidden_layer.spike_trains
            errors = self.ttfs_mse(spike_per_neuron, spike_targets)

            weights_gradient, _ = hidden_layer.backward(errors)

            delta_weights = -learning_rate * cp.mean(weights_gradient, axis=0)

            hidden_layer.add_deltas(delta_weights)

        self.assertTrue(np.allclose(spike_per_neuron[:, :, 0].get(), spike_targets, atol=5e-4))

    def test_store_restore(self):
        n_pre = 7
        n_post = 5
        tau_s = 10e-3
        theta = 0.01
        delta_theta = theta
        max_n_spike = 8
        weights = cp.random.normal(0, 1, (n_post, n_pre))
        bias = cp.random.normal(0, 1, (n_post,))

        input_layer = InputLayer(n_neurons=n_pre)
        hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=n_post, tau_s=tau_s, theta=theta,
                                delta_theta=delta_theta,
                                weight_initializer=lambda n_post, n_pre: weights, max_n_spike=max_n_spike,
                                name="test_layer")

        save_dir = Path("/tmp/bats_test_store_restore")
        save_dir.mkdir(parents=True, exist_ok=True)
        hidden_layer.store(save_dir)

        del hidden_layer

        new_hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=n_post, tau_s=tau_s, theta=theta,
                                    delta_theta=delta_theta, max_n_spike=max_n_spike,
                                    name="test_layer")
        new_hidden_layer.restore(save_dir)

        self.assertTrue(np.allclose(new_hidden_layer.weights, weights))