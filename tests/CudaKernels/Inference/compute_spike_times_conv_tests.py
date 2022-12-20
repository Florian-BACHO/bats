import unittest
from brian2 import *
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import compute_spike_times_conv, get_spike_weights


class CudaKernelsTests(unittest.TestCase):
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
        return spike_monitor.all_values()

    def idx_to_pos(self, idx, shape):
        tmp = shape[1] * shape[2]
        x = idx / tmp
        tmp = idx % tmp
        y = tmp / shape[2]
        z = tmp % shape[2]
        return int(x), int(y), int(z)

    def test_compute_spike_times(self):
        batch_size = 2
        pre_shape = np.array([10, 10, 3])
        pre_shape_gpu = cp.array(pre_shape, dtype=cp.int32)
        filter_shape = np.array([2, 5, 5, 3])
        filter_shape_gpu = cp.array(filter_shape, dtype=cp.int32)
        post_shape = np.array([pre_shape[0] - filter_shape[1] + 1,
                               pre_shape[1] - filter_shape[2] + 1,
                               filter_shape[0]])
        post_shape_gpu = cp.array(post_shape, dtype=cp.int32)
        n_pre = np.prod(pre_shape)
        n_post = np.prod(post_shape)
        n_spike = 100
        tau_s = np.float32(100e-3)
        tau = np.float32(2 * tau_s)
        theta = 1e-4
        delta_theta = 0.2
        theta_tau = np.float32(theta / tau)
        delta_theta_tau = np.float32(delta_theta / tau)
        max_n_spike = np.int32(4)
        max_simulation = 500e-3

        np.random.seed(42)
        spike_indices_cpu = np.random.randint(n_pre, size=(batch_size, n_spike))
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32))

        spike_times_cpu = np.sort(np.random.uniform(low=0.0, high=max_simulation, size=(batch_size, n_spike)), axis=1)
        spike_times_cpu[0, -28:] = np.inf
        spike_times_cpu[0, -10:] = np.inf
        spike_times_cpu[0, -3:] = np.inf
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32))

        weights_cpu = np.random.normal(loc=0, scale=1.0, size=filter_shape)
        weights_gpu = cp.array(weights_cpu.astype(np.float32))

        exp_tau_s, exp_tau = cp.exp(spike_times_gpu / tau_s), cp.exp(spike_times_gpu / tau)
        c = theta_tau

        n_spikes, _, _, spike_times, _ = \
            compute_spike_times_conv(spike_indices_gpu, spike_times_gpu, exp_tau_s, exp_tau,
                                     weights_gpu, c, delta_theta_tau,
                                     tau, cp.float32(max_simulation), max_n_spike, pre_shape_gpu, post_shape_gpu,
                                     filter_shape_gpu)

        weights_brian = np.zeros((n_post, n_pre))
        for post_idx in range(n_post):
            post_x, post_y, post_c = self.idx_to_pos(post_idx, post_shape)
            for pre_idx in range(n_pre):
                pre_x, pre_y, pre_c = self.idx_to_pos(pre_idx, pre_shape)
                # Check connection
                if pre_x < post_x or pre_x >= (post_x + filter_shape[1]) or \
                    pre_y < post_y or pre_y >= (post_y + filter_shape[2]):
                    continue
                pos_x = pre_x - post_x
                pos_y = pre_y - post_y
                weights_brian[post_idx, pre_idx] = weights_cpu[post_c, pos_x, pos_y, pre_c]

        # Infer brian
        true_spike_times = []
        for indices, times in zip(spike_indices_cpu, spike_times_cpu):
            tmp = self.compute_spike_times_brian2(n_pre, indices, times, n_post, tau_s,
                                                  tau, theta, delta_theta, weights_brian,
                                                  max_simulation)
            true_spike_times.append(tmp)

        # For each batch
        for times, n_s, true_ti in zip(spike_times, n_spikes, true_spike_times):
            # For each neuron spike train
            for i, true_times in true_ti['t'].items():
                n = n_s.get()[i]
                t = times.get()[i][:n]
                true_times = np.sort(true_times / second)
                if len(true_times) > max_n_spike:
                    true_times = true_times[:max_n_spike]
                self.assertTrue(n == len(true_times))
                self.assertTrue(np.allclose(true_times, t, atol=1e-2))
