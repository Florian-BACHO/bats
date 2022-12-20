from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Inference/get_spike_weights.cu"
KERNEL_NAME = "get_spike_weights_kernel"

__get_spike_weights_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def get_spike_weights(weights: cp.ndarray, spike_indices: cp.ndarray) -> cp.ndarray:
    batch_size, max_n_pre_spike = spike_indices.shape
    n_neurons, n_weight_per_neuron = weights.shape
    block_dim = (batch_size, 1, 1)
    grid_dim = (max_n_pre_spike, n_neurons, 1)

    out = cp.ndarray((batch_size, n_neurons, max_n_pre_spike), dtype=cp.float32)
    __get_spike_weights_kernel(grid_dim, block_dim, (weights, spike_indices, out, cp.int32(n_weight_per_neuron)))

    return out