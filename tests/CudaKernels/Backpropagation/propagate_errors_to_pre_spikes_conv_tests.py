import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import propagate_errors_to_pre_spikes, \
    propagate_errors_to_pre_spikes_conv


class CudaKernelsTests(unittest.TestCase):
    def idx_to_pos(self, idx, shape):
        tmp = shape[1] * shape[2]
        x = idx / tmp
        tmp = idx % tmp
        y = tmp / shape[2]
        z = tmp % shape[2]
        return int(x), int(y), int(z)

    def propagate_errors_to_pre_spikes(self, f1, f2, post_spike_times, pre_spike_times,
                                       exp_tau_s, exp_tau, weights, errors, tau_s, tau,
                                       pre_shape, post_shape, filter_shape):
        out = np.zeros(exp_tau_s.shape)
        for s_out, s_f1, s_f2, s_post_spike_times, s_pre_spike_times, s_exp_tau_s, s_exp_tau, s_errors in \
                zip(out, f1, f2, post_spike_times, pre_spike_times, exp_tau_s, exp_tau, errors):
            for pre_idx, (n_out, n_pre_spike_times, n_exp_tau_s, n_exp_tau) in \
                    enumerate(zip(s_out, s_pre_spike_times, s_exp_tau_s, s_exp_tau)):
                pre_x, pre_y, pre_c = self.idx_to_pos(pre_idx, pre_shape)
                for post_idx, (n_f1, n_f2, n_post_t, n_weights, n_errors) in \
                        enumerate(zip(s_f1, s_f2, s_post_spike_times, weights, s_errors)):
                    post_x, post_y, post_c = self.idx_to_pos(post_idx, post_shape)
                    # Check connection
                    if pre_x < post_x or pre_x >= (post_x + filter_shape[1]) or \
                            pre_y < post_y or pre_y >= (post_y + filter_shape[2]):
                        continue
                    for i, (pre_spike_t, spike_exp_tau_s, spike_exp_tau) in \
                            enumerate(zip(n_pre_spike_times, n_exp_tau_s, n_exp_tau)):
                        if np.isinf(pre_spike_t):
                            break
                        w = n_weights[pre_idx]
                        for spike_f1, spike_f2, post_spike_t, spike_error in \
                                zip(n_f1, n_f2, n_post_t, n_errors):
                            if np.isinf(post_spike_t):
                                break
                            if post_spike_t <= pre_spike_t:
                                continue
                            n_out[i] += spike_error * w * (spike_f1 * spike_exp_tau_s / tau_s -
                                                           spike_f2 * spike_exp_tau / tau)
        return out

    def test_propagate_errors_to_pre_spikes_conv(self):
        tau_s = 0.1
        tau = 2 * tau_s
        batch_size = 5
        pre_shape = np.array([5, 5, 3])
        pre_shape_gpu = cp.array(pre_shape, dtype=cp.int32)
        filter_shape = np.array([4, 5, 5, 3])
        filter_shape_gpu = cp.array(filter_shape, dtype=cp.int32)
        post_shape = np.array([pre_shape[0] - filter_shape[1] + 1,
                               pre_shape[1] - filter_shape[2] + 1,
                               filter_shape[0]])
        post_shape_gpu = cp.array(post_shape, dtype=cp.int32)
        n_pre = np.prod(pre_shape)
        n_post = np.prod(post_shape)
        max_n_pre_spike = 4
        max_n_post_spike = 3

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

        pre_errors = propagate_errors_to_pre_spikes_conv(f1_gpu, f2_gpu, post_spike_times_gpu, pre_spike_times_gpu,
                                                    exp_tau_s_gpu, exp_tau_gpu, weights_gpu, errors_gpu,
                                                    cp.float32(tau_s), cp.float32(tau),
                                                    pre_shape_gpu, post_shape_gpu, filter_shape_gpu)

        true_pre_errors = self.propagate_errors_to_pre_spikes(f1, f2, post_spike_times, pre_spike_times,
                                                              exp_tau_s, exp_tau, weights, errors, tau_s, tau,
                                                              pre_shape, post_shape, filter_shape)

        np.testing.assert_allclose(pre_errors.get(), true_pre_errors, atol=1e-1)
