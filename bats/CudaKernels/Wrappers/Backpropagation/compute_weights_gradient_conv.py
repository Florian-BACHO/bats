import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_weights_gradient_conv.cu"
KERNEL_NAME = "compute_weights_gradient_conv_kernel"

__compute_weights_gradient_conv_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def compute_weights_gradient_conv(f1: cp.ndarray, f2: cp.ndarray,
                                  post_times: cp.ndarray, pre_times: cp.array,
                                  pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                                  pre_shape: cp.ndarray, post_shape: cp.ndarray,
                                  filter_shape: cp.ndarray,
                                  errors: cp.ndarray) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    filter_c, filter_x, filter_y, filter_z = filter_shape.get()
    gradient = cp.zeros((batch_size, filter_c, filter_x, filter_y, filter_z), dtype=cp.float32)

    block_dim = (batch_size, 1, 1)
    grid_dim = (filter_x, filter_y, filter_z)
    __compute_weights_gradient_conv_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                                 pre_exp_tau_s, pre_exp_tau, errors,
                                                                 pre_shape, post_shape, filter_shape,
                                                                 gradient,
                                                                 n_post_neurons, n_pre_neurons,
                                                                 cp.int32(max_n_post_spike),
                                                                 cp.int32(max_n_pre_spike)))
    return gradient
