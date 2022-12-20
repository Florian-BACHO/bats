import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import compute_pre_exps

class CudaKernelsTests(unittest.TestCase):
    def test_compute_pre_exps(self):
        batch_size = 100
        n_neurons = 100
        max_n_spike = 10
        tau_s = np.float32(0.1)
        tau = np.float32(2 * tau_s)

        spike_times_cpu = np.sort(np.random.uniform(low=0.0, high=0.3, size=(batch_size, n_neurons, max_n_spike)), axis=1)
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32), dtype=cp.float32)

        exp_tau_s, exp_tau = compute_pre_exps(spike_times_gpu, cp.float32(tau_s), cp.float32(tau))
        true_exp_tau_s = np.exp(spike_times_cpu / tau_s)
        true_exp_tau = np.exp(spike_times_cpu / tau)

        self.assertTrue(np.allclose(true_exp_tau_s, exp_tau_s.get()) and
                        np.allclose(true_exp_tau, exp_tau.get()))