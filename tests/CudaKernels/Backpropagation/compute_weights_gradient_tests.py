import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import compute_weights_gradient


class CudaKernelsTests(unittest.TestCase):
    def compute_weights_gradient_cpu(self, f1, f2, post_times, pre_times, exp_tau_s, exp_tau, errors):
        batch_size, n_post_neurons, max_n_post_spike = f1.shape
        _, n_pre_neurons, max_n_pre_spike = exp_tau_s.shape
        out = np.zeros((batch_size, n_post_neurons, n_pre_neurons))

        for s_out, s_f1, s_f2, s_post_times, s_pre_times, s_exp_tau_s, s_exp_tau, s_errors in \
            zip(out, f1, f2, post_times, pre_times, exp_tau_s, exp_tau, errors):
            for n_out, n_f1, n_f2, n_post_times, n_errors in zip(s_out, s_f1, s_f2, s_post_times, s_errors):
                for i, (spike_f1, spike_f2, time, spike_error) in enumerate(zip(n_f1, n_f2, n_post_times, n_errors)):
                    for n_pre, (pre_t, e_t_s, e_t) in enumerate(zip(s_pre_times, s_exp_tau_s, s_exp_tau)):
                        mask = pre_t < time
                        exp_tau_s_sum = np.sum(e_t_s[mask])
                        exp_tau_sum = np.sum(e_t[mask])
                        n_out[n_pre] += spike_error * (spike_f1 * exp_tau_s_sum - spike_f2 * exp_tau_sum)

        return out

    def test_compute_weights_gradient(self):
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

        grad = compute_weights_gradient(f1_gpu, f2_gpu, post_spike_times_gpu, pre_spike_times_gpu,
                                        exp_tau_s_gpu, exp_tau_gpu, errors_gpu)
        true_grad = self.compute_weights_gradient_cpu(f1, f2, post_spike_times, pre_spike_times,
                                                      exp_tau_s, exp_tau, errors)

        self.assertTrue(np.allclose(grad, true_grad, atol=1e-1))