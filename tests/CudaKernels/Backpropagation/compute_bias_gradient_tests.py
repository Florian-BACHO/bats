import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import compute_bias_gradient


class CudaKernelsTests(unittest.TestCase):
    def compute_bias_gradient_cpu(self, spike_times, f2, f3, errors, bias_scale):
        batch_size, n_post_neurons, max_n_post_spike = f2.shape
        out = np.zeros((batch_size, n_post_neurons))

        for s_out, s_times, s_f2, s_f3, s_errors in zip(out, spike_times, f2, f3, errors):
            for i, (n_times, n_f2, n_f3, n_errors) in enumerate(zip(s_times, s_f2, s_f3, s_errors)):
                mask = np.isfinite(n_times)
                s_out[i] = np.sum(n_errors * bias_scale * (n_f2[mask] - n_f3[mask]))

        return out

    def test_compute_bias_gradient(self):
        batch_size = 100
        n_post = 50
        max_n_post_spike = 30
        bias_scale = 5
        f2 = np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike))
        f2_gpu = cp.array(f2, dtype=cp.float32)
        f3 = np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike))
        f3_gpu = cp.array(f3, dtype=cp.float32)
        spike_times = np.sort(np.random.uniform(0, 1.0, (batch_size, n_post, max_n_post_spike)), axis=2)
        spike_times_gpu = cp.array(spike_times, dtype=cp.float32)
        errors = np.random.uniform(-1.0, 1.0, (batch_size, n_post, max_n_post_spike))
        errors_gpu = cp.array(errors, dtype=cp.float32)

        grad = compute_bias_gradient(spike_times_gpu, f2_gpu, f3_gpu, errors_gpu, bias_scale)
        true_grad = self.compute_bias_gradient_cpu(spike_times, f2, f3, errors, bias_scale)

        self.assertTrue(np.allclose(grad.get(), true_grad, atol=1e-4))
