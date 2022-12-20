import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import compute_weights_gradient, compute_weights_gradient_conv


class CudaKernelsTests(unittest.TestCase):
    def idx_to_pos(self, idx, shape):
        tmp = shape[1] * shape[2]
        x = idx / tmp
        tmp = idx % tmp
        y = tmp / shape[2]
        z = tmp % shape[2]
        return int(x), int(y), int(z)

    def compute_weights_gradient_conv_cpu(self, f1, f2, post_times, pre_times, exp_tau_s, exp_tau,
                                          pre_shape, post_shape, filter_shape, errors):
        batch_size, n_post_neurons, max_n_post_spike = f1.shape
        _, n_pre_neurons, max_n_pre_spike = exp_tau_s.shape
        out = np.zeros((batch_size, filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]))

        # Each batch
        for sample, (s_f1, s_f2, s_post_times, s_pre_times, s_exp_tau_s, s_exp_tau, s_errors) in \
                enumerate(zip(f1, f2, post_times, pre_times, exp_tau_s, exp_tau, errors)):
            # Each post_neuron
            for post_idx, (n_f1, n_f2, n_post_times, n_errors) in enumerate(zip(s_f1, s_f2, s_post_times, s_errors)):
                post_x, post_y, post_c = self.idx_to_pos(post_idx, post_shape)
                # Each spike
                for spike_f1, spike_f2, time, spike_error in zip(n_f1, n_f2, n_post_times, n_errors):
                    # Each pre-neuron
                    for pre_idx, (pre_t, e_t_s, e_t) in enumerate(zip(s_pre_times, s_exp_tau_s, s_exp_tau)):
                        pre_x, pre_y, pre_c = self.idx_to_pos(pre_idx, pre_shape)
                        if pre_x < post_x or pre_x >= (post_x + filter_shape[1]) or \
                                pre_y < post_y or pre_y >= (post_y + filter_shape[2]):
                            continue
                        mask = pre_t < time
                        exp_tau_s_sum = np.sum(e_t_s[mask])
                        exp_tau_sum = np.sum(e_t[mask])
                        pos_x = pre_x - post_x
                        pos_y = pre_y - post_y
                        out[sample, post_c, pos_x, pos_y, pre_c] += spike_error * (spike_f1 * exp_tau_s_sum -
                                                                                   spike_f2 * exp_tau_sum)

        return out

    def test_compute_weights_gradient_conv(self):
        tau_s = 0.1
        tau = 2 * tau_s
        batch_size = 5
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

        max_n_pre_spike = 5
        max_n_post_spike = 4
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

        grad = compute_weights_gradient_conv(f1_gpu, f2_gpu, post_spike_times_gpu, pre_spike_times_gpu,
                                             exp_tau_s_gpu, exp_tau_gpu, pre_shape_gpu, post_shape_gpu,
                                             filter_shape_gpu, errors_gpu)

        true_grad = self.compute_weights_gradient_conv_cpu(f1, f2, post_spike_times, pre_spike_times,
                                                           exp_tau_s, exp_tau, pre_shape, post_shape, filter_shape,
                                                           errors)

        self.assertTrue(np.allclose(grad, true_grad, atol=1e-1))
