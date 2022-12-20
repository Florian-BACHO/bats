from typing import Tuple
from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Inference/compute_pre_exps.cu"
KERNEL_NAME = "compute_pre_exps_kernel"

__compute_pre_exps_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def compute_pre_exps(spike_times: cp.ndarray, tau_s: cp.float32, tau: cp.float32) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size, n_neurons, max_n_spike = spike_times.shape
    block_dim = (batch_size, 1, 1)
    grid_dim = (max_n_spike, n_neurons, 2)

    exp_tau_s = cp.ndarray(spike_times.shape, dtype=cp.float32)
    exp_tau = cp.ndarray(spike_times.shape, dtype=cp.float32)
    __compute_pre_exps_kernel(grid_dim, block_dim, (spike_times, exp_tau_s, exp_tau, tau_s, tau))
    return exp_tau_s, exp_tau