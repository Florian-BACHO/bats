import cupy as cp
import numpy as np

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_weights_gradient.cu"
KERNEL_NAME = "compute_weights_gradient_kernel"

__compute_weights_gradient_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

#! This causes a problem I guess
def compute_weights_gradient(f1: cp.ndarray, f2: cp.ndarray,
                             post_times: cp.ndarray, pre_times: cp.array,
                             pre_exp_tau_s: cp.ndarray, pre_exp_tau: cp.ndarray,
                             errors: cp.ndarray) -> cp.ndarray:
    batch_size, n_post_neurons, max_n_post_spike = f1.shape
    _, n_pre_neurons, max_n_pre_spike = pre_exp_tau_s.shape

    gradient = cp.zeros((batch_size, n_post_neurons, n_pre_neurons), dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (n_post_neurons, n_pre_neurons)
    #! Check for NaNs
    if np.isnan(gradient).any():
        print("gradient has nan values")
    if cp.isnan(f1).any():
        print("f1 has nan values")
    if cp.isnan(f2).any():
        print("f2 has nan values")
    if cp.isnan(post_times).any():
        print("post_times has nan values")
    if cp.isnan(pre_times).any():
        print("pre_times has nan values")
    if cp.isnan(pre_exp_tau_s).any():
        print("pre_exp_tau_s has nan values")
    if cp.isnan(pre_exp_tau).any():
        print("pre_exp_tau has nan values")
    if cp.isnan(errors).any():
        print("errors has nan values")
    __compute_weights_gradient_kernel(grid_dim, block_dim, (f1, f2, post_times, pre_times,
                                                            pre_exp_tau_s, pre_exp_tau, errors,
                                                            gradient, cp.int32(max_n_post_spike),
                                                            cp.int32(max_n_pre_spike)))
    return gradient