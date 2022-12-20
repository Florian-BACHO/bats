import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import propagate_errors_to_pre_spikes


class CudaKernelsTests(unittest.TestCase):
    def propagate_errors_to_pre_spikes(self, f1, f2, post_spike_times, pre_spike_times,
                                       exp_tau_s, exp_tau, weights, errors, tau_s, tau):
        out = np.zeros(exp_tau_s.shape)
        for s_out, s_f1, s_f2, s_post_spike_times, s_pre_spike_times, s_exp_tau_s, s_exp_tau, s_errors in \
                zip(out, f1, f2, post_spike_times, pre_spike_times, exp_tau_s, exp_tau, errors):
            for i, (n_out, n_pre_spike_times, n_exp_tau_s, n_exp_tau) in \
                    enumerate(zip(s_out, s_pre_spike_times, s_exp_tau_s, s_exp_tau)):
                for j, (pre_spike_t, spike_exp_tau_s, spike_exp_tau) in \
                        enumerate(zip(n_pre_spike_times, n_exp_tau_s, n_exp_tau)):
                    if np.isinf(pre_spike_t):
                        break
                    for n_f1, n_f2, n_post_t, n_weights, n_errors in \
                            zip(s_f1, s_f2, s_post_spike_times, weights, s_errors):
                        w = n_weights[i]
                        for spike_f1, spike_f2, post_spike_t, spike_error in \
                                zip(n_f1, n_f2, n_post_t, n_errors):
                            if np.isinf(post_spike_t):
                                break
                            if post_spike_t <= pre_spike_t:
                                continue
                            n_out[j] += spike_error * w * (spike_f1 * spike_exp_tau_s / tau_s -
                                                           spike_f2 * spike_exp_tau / tau)
        return out

    def test_propagate_errors_to_pre_spikes(self):
        tau_s = 0.1
        tau = 2 * tau_s
        batch_size = 10
        n_pre = 20
        n_post = 10
        max_n_pre_spike = 25
        max_n_post_spike = 12
        f1 = np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike))
        f1_gpu = cp.array(f1, dtype=cp.float32)
        f2 = np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike))
        f2_gpu = cp.array(f2, dtype=cp.float32)
        pre_spike_times = np.sort(np.random.uniform(0, 1.0, (batch_size, n_pre, max_n_pre_spike)), axis=2)
        pre_spike_times_gpu = cp.array(pre_spike_times, dtype=cp.float32)
        post_spike_times = np.sort(np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike)), axis=2)
        post_spike_times_gpu = cp.array(post_spike_times, dtype=cp.float32)
        exp_tau_s = np.exp(pre_spike_times / tau_s)
        exp_tau_s_gpu = cp.array(exp_tau_s, dtype=cp.float32)
        exp_tau = np.exp(pre_spike_times / tau)
        exp_tau_gpu = cp.array(exp_tau, dtype=cp.float32)
        errors = np.random.uniform(-1.0, 1.0, (batch_size, n_post, max_n_post_spike))
        errors_gpu = cp.array(errors, dtype=cp.float32)
        weights = np.random.normal(loc=0, scale=1.0, size=(n_post, n_pre))
        weights_gpu = cp.array(weights, dtype=cp.float32)

        pre_errors = propagate_errors_to_pre_spikes(f1_gpu, f2_gpu, post_spike_times_gpu, pre_spike_times_gpu,
                                                    exp_tau_s_gpu, exp_tau_gpu, weights_gpu, errors_gpu,
                                                    cp.float32(tau_s), cp.float32(tau))
        true_pre_errors = self.propagate_errors_to_pre_spikes(f1, f2, post_spike_times, pre_spike_times,
                                                              exp_tau_s, exp_tau, weights, errors, tau_s, tau)

        self.assertTrue(np.allclose(pre_errors, true_pre_errors, atol=1e-1))
