import unittest
from brian2 import *
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import compute_spike_times, get_spike_weights


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

    def test_compute_spike_times(self):
        batch_size = 1
        n_pre = 10
        n_post = 5
        n_spike = 100
        tau_s = np.float32(100e-3)
        tau = np.float32(2 * tau_s)
        theta = 0.1
        delta_theta = 0.2
        theta_tau = np.float32(theta / tau)
        delta_theta_tau = np.float32(delta_theta / tau)
        max_n_spike = np.int32(128)
        max_simulation = 500e-3

        np.random.seed(42)
        spike_indices_cpu = np.random.randint(n_pre, size=(batch_size, n_spike))
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32))

        spike_times_cpu = np.sort(np.random.uniform(low=0.0, high=max_simulation, size=(batch_size, n_spike)), axis=1)
        spike_times_cpu[0, -28:] = np.inf
        spike_times_cpu[0, -10:] = np.inf
        spike_times_cpu[0, -3:] = np.inf
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32))

        weights_cpu = np.random.normal(loc=0, scale=1.0, size=(n_post, n_pre))
        weights_gpu = cp.array(weights_cpu.astype(np.float32))

        exp_tau_s, exp_tau = cp.exp(spike_times_gpu / tau_s), cp.exp(spike_times_gpu / tau)
        spike_weights_gpu = get_spike_weights(weights_gpu, spike_indices_gpu)
        c = theta_tau

        n_spikes, _, _, spike_times, _ = \
            compute_spike_times(spike_times_gpu, exp_tau_s, exp_tau,
                                spike_weights_gpu, c, delta_theta_tau,
                                tau, cp.float32(max_simulation), max_n_spike)

        true_spike_times = []
        for indices, times in zip(spike_indices_cpu, spike_times_cpu):
            tmp = self.compute_spike_times_brian2(n_pre, indices, times, n_post, tau_s,
                                                  tau, theta, delta_theta, weights_cpu,
                                                  max_simulation)
            true_spike_times.append(tmp)

        for times, n_s, true_ti in zip(spike_times, n_spikes, true_spike_times):
            for i, true_times in true_ti['t'].items():
                n = n_s.get()[i]
                t = times.get()[i][:n]
                self.assertTrue(n == len(true_times))
                self.assertTrue(np.allclose(np.sort(true_times / second), t, atol=1e-2))