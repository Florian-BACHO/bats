import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import convert_to_spike_per_neuron


class CudaKernelsTests(unittest.TestCase):
    def test_convert_to_spike_per_neuron(self):
        n_neurons = 3
        spike_indices_cpu = np.array([[0, 0, 2, 2, 0, 1],
                                      [2, 2, 0, 1, -1, -1],
                                      [1, -1, -1, -1, -1, -1]])
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32))

        spike_times_cpu = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                    [0.25, 0.3, 0.45, 0.57, np.inf, np.inf],
                                    [0.33, np.inf, np.inf, np.inf, np.inf, np.inf]])
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32))

        true_spike_per_neuron = np.array([[[0.1, 0.2, 0.5],
                                           [0.6, np.inf, np.inf],
                                           [0.3, 0.4, np.inf]],
                                          [[0.45, np.inf, np.inf],
                                           [0.57, np.inf, np.inf],
                                           [0.25, 0.3, np.inf]],
                                          [[np.inf, np.inf, np.inf],
                                           [0.33, np.inf, np.inf],
                                           [np.inf, np.inf, np.inf]]])
        true_n_spike_per_neuron = np.array([[3, 1, 2],
                                            [1, 1, 2],
                                            [0, 1, 0]])
        spike_per_neuron, n_spike_per_neuron = convert_to_spike_per_neuron(spike_indices_gpu, spike_times_gpu,
                                                                           cp.int32(n_neurons))

        self.assertTrue((true_n_spike_per_neuron == n_spike_per_neuron.get()).all())
        self.assertTrue(np.allclose(true_spike_per_neuron, spike_per_neuron))