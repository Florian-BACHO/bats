import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Inference import convert_to_spike_per_neuron, get_sorted_spikes_indices


class CudaKernelsTests(unittest.TestCase):
    def test_get_sorted_spikes_indices(self):
        batch_size = 100
        n_neurons = 50
        n_spikes = 200
        spike_indices_cpu = np.random.randint(0, n_neurons, size=(batch_size, n_spikes))
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32))

        spike_times_cpu = np.sort(np.random.uniform(0, 1.0, size=(batch_size, n_spikes)), axis=1)
        spike_times_cpu[0, -30:] = np.inf
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32))

        spike_per_neuron, n_spike_per_neuron = convert_to_spike_per_neuron(spike_indices_gpu, spike_times_gpu,
                                                                           cp.int32(n_neurons))
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(spike_per_neuron,
                                                                                    n_spike_per_neuron)
        sorted_spike_indices = sorted_indices // spike_per_neuron.shape[2]
        sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
        mask = cp.isfinite(sorted_spike_times).get()
        self.assertTrue((sorted_spike_indices.get()[mask] == spike_indices_cpu[mask]).all())
        self.assertTrue(np.allclose(sorted_spike_times, spike_times_cpu))