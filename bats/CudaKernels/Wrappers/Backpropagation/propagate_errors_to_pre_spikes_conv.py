import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/propagate_errors_to_pre_spikes_conv.cu"
KERNEL_NAME = "propagate_errors_to_pre_spikes_conv_kernel"

__propagate_errors_to_pre_spikes_conv_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def propagate_errors_to_pre_spikes_conv(f1: cp.ndarray, f2: cp.ndarray,
                                        post_times: cp.ndarray, pre_times: cp.array,
                                        pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                                        weights: cp.ndarray,
                                        errors: cp.ndarray, tau_s: cp.float32, tau: cp.float32,
                                        pre_shape: cp.ndarray, post_shape: cp.ndarray,
                                        filter_shape: cp.ndarray) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    n_post_neurons_filter = cp.prod(filter_shape[:-1]).get()
    pre_errors = cp.zeros((batch_size, n_pre_neurons, max_n_pre_spike, n_post_neurons_filter), dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (n_pre_neurons, max_n_pre_spike, n_post_neurons_filter)
    __propagate_errors_to_pre_spikes_conv_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                       pre_exp_tau_s, pre_exp_tau, weights, errors,
                                                                       post_shape, pre_shape, filter_shape,
                                                                       pre_errors, cp.int32(n_post_neurons),
                                                                       cp.int32(max_n_post_spike),
                                                                       tau_s, tau))
    pre_errors = cp.sum(pre_errors, axis=3)
    return pre_errors
