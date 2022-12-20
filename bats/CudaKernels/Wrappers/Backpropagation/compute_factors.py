from typing import Tuple
import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_factors.cu"
KERNEL_NAME = "compute_factors_kernel"

__compute_factors_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def compute_factors(spike_times: cp.array, a: cp.ndarray, c: cp.float32, x: cp.ndarray, exp_tau: cp.ndarray,
                    tau: cp.float32) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size, n_neurons, max_n_spike = a.shape

    f1 = cp.ndarray(a.shape, dtype=cp.float32)
    f2 = cp.ndarray(a.shape, dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (max_n_spike, n_neurons, 2)

    __compute_factors_kernel(grid_dim, block_dim, (spike_times, a, c, x, exp_tau, tau,
                                                   f1, f2))
    return f1, f2
