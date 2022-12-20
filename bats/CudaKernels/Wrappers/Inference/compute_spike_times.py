from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp
import numpy as np

KERNEL_FILE = "Inference/compute_spike_times.cu"
KERNEL_NAME = "compute_spike_times_kernel"

__compute_spike_times_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)


def compute_spike_times(spike_times: cp.ndarray,
                        exp_tau_s: cp.ndarray, exp_tau: cp.ndarray,
                        spike_weights: cp.ndarray,
                        c: cp.float32, delta_theta_tau: np.float32, tau: np.float32,
                        max_simulation: np.float32, max_n_post_spikes: np.int32):
    batch_size, n_neurons, max_n_pre_spike = spike_weights.shape
    block_dim = (batch_size, 1, 1)
    grid_dim = (n_neurons, 1, 1)

    res_shape = (batch_size, n_neurons, max_n_post_spikes)
    n_spikes = cp.zeros((batch_size, n_neurons), dtype=cp.int32)
    a = cp.ndarray(res_shape, dtype=cp.float32)
    x = cp.ndarray(res_shape, dtype=cp.float32)
    post_spike_times = cp.full(res_shape, cp.inf, dtype=cp.float32)
    post_exp_tau = cp.full(res_shape, cp.inf, dtype=cp.float32)

    args = (spike_times, exp_tau_s, exp_tau, spike_weights, c, delta_theta_tau, tau,
            max_simulation, max_n_pre_spike, max_n_post_spikes,
            n_spikes, a, x, post_spike_times, post_exp_tau)
    __compute_spike_times_kernel(grid_dim, block_dim, args)

    return n_spikes, a, x, post_spike_times, post_exp_tau
