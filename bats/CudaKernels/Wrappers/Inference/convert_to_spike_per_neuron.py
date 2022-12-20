from typing import Tuple

from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Inference/convert_to_spike_per_neuron.cu"
KERNEL_NAME = "convert_to_spike_per_neuron_kernel"

__convert_to_spike_per_neuron_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def __get_max_spike(indices):
    out = cp.zeros((1,), dtype=cp.int32)
    for it in indices:
        values, counts = cp.unique(it, return_counts=True)
        sample_max = cp.max(counts[1:]) if values[0] == -1 else cp.max(counts)
        if sample_max > out:
            out = sample_max
    return cp.int32(out.get())

def convert_to_spike_per_neuron(indices: cp.ndarray, times: cp.ndarray, n_neurons: cp.int32) \
        -> Tuple[cp.ndarray, cp.ndarray]:
    max_spike_per_neuron = __get_max_spike(indices)
    batch_size, max_pre_spike = indices.shape

    times_per_neuron = cp.full((batch_size, n_neurons, max_spike_per_neuron), cp.inf, dtype=cp.float32)
    n_per_neuron = cp.zeros((batch_size, n_neurons), dtype=cp.int32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (1, 1, 1)

    __convert_to_spike_per_neuron_kernel(grid_dim, block_dim, (indices, times, times_per_neuron, n_per_neuron,
                                                               n_neurons, max_pre_spike, max_spike_per_neuron))

    return times_per_neuron, n_per_neuron