import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_weights_gradient_old.cu"
KERNEL_NAME = "compute_weights_gradient_kernel"

__compute_weights_gradient_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def compute_weights_gradient(f1: cp.ndarray, f2: cp.ndarray,
                             post_times: cp.ndarray, pre_times: cp.array,
                             pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                             errors: cp.ndarray) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    spikes_gradient = cp.zeros((batch_size, n_post_neurons, max_n_post_spike, n_pre_neurons), dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (n_post_neurons, max_n_post_spike, n_pre_neurons)
    __compute_weights_gradient_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                            pre_exp_tau_s, pre_exp_tau, errors,
                                                            spikes_gradient, cp.int32(max_n_pre_spike)))
    spikes_gradient = cp.sum(spikes_gradient, axis=2)
    return spikes_gradient