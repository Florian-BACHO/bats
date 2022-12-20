import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import compute_factors


class CudaKernelsTests(unittest.TestCase):
    def compute_f1_f2(self, a, c, x, exp_tau, tau):
        f1 = tau / a * (1 + c / x * exp_tau)
        f2 = tau / x
        return f1, f2

    def test_compute_factors(self):
        thresh = 1.0
        tau = 0.2
        batch_size = 100
        n_neurons = 50
        max_n_spike = 100
        a = np.random.uniform(0, 10.0, (batch_size, n_neurons, max_n_spike))
        a_gpu = cp.array(a, dtype=cp.float32)
        x = np.random.uniform(0, 10.0, (batch_size, n_neurons, max_n_spike))
        x_gpu = cp.array(x, dtype=cp.float32)
        c = thresh / tau
        spike_times = np.sort(np.random.uniform(0, 10.0, (batch_size, n_neurons, max_n_spike)), axis=1)
        spike_times_gpu = cp.array(spike_times, dtype=cp.float32)
        exp_tau = np.exp(spike_times / tau)
        exp_tau_gpu = cp.array(exp_tau, dtype=cp.float32)
        f1, f2 = compute_factors(spike_times_gpu, a_gpu, cp.float32(c), x_gpu, exp_tau_gpu, cp.float32(tau))
        true_f1, true_f2 = self.compute_f1_f2(a, c, x, exp_tau, tau)

        self.assertTrue(np.allclose(f1.get(), true_f1, rtol=1e-3))
        self.assertTrue(np.allclose(f2.get(), true_f2))