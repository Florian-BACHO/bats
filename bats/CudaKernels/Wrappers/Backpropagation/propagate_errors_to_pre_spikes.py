import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/propagate_errors_to_pre_spikes.cu"
KERNEL_NAME = "propagate_errors_to_pre_spikes_kernel"

__propagate_errors_to_pre_spikes_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def propagate_errors_to_pre_spikes(f1: cp.ndarray, f2: cp.ndarray,
                                   post_times: cp.ndarray, pre_times: cp.array,
                                   pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                                   weights: cp.ndarray,
                                   errors: cp.ndarray, tau_s: cp.float32, tau: cp.float32) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    pre_errors = cp.zeros((batch_size, n_pre_neurons, max_n_pre_spike, n_post_neurons), dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (n_pre_neurons, max_n_pre_spike, n_post_neurons)
    __propagate_errors_to_pre_spikes_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                  pre_exp_tau_s, pre_exp_tau, weights, errors,
                                                                  pre_errors, cp.int32(max_n_post_spike),
                                                                  tau_s, tau))
    pre_errors = cp.sum(pre_errors, axis=3)
    return pre_errors
