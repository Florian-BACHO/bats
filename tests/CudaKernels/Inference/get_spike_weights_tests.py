import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import get_spike_weights


class CudaKernelsTests(unittest.TestCase):
    def get_spike_weight_cpu(self, weights: np.ndarray, spike_indices: np.ndarray) -> np.ndarray:
        n_neurons, n_spike = weights.shape
        out = []
        for sample_indices in spike_indices:
            sample_weights = []
            for n in range(n_neurons):
                neuron_weights = weights[n]
                sample_weights.append(neuron_weights[sample_indices])
            out.append(np.stack(sample_weights, axis=0))
        return np.stack(out, axis=0)

    def test_get_spike_weights(self):
        batch_size = 10
        n_pre = 1000
        n_post = 500
        n_spike = 2000

        spike_indices_cpu = np.random.randint(n_pre, size=(batch_size, n_spike))
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32), dtype=cp.int32)
        weights_cpu = np.random.normal(loc=0, scale=1.0, size=(n_post, n_pre))
        weights_gpu = cp.array(weights_cpu.astype(np.float32), dtype=cp.float32)

        spike_weights_gpu = get_spike_weights(weights_gpu, spike_indices_gpu)
        true_spike_weights = self.get_spike_weight_cpu(weights_cpu, spike_indices_cpu)

        self.assertTrue(np.allclose(spike_weights_gpu.get(), true_spike_weights))