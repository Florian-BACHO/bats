from typing import Tuple
import cupy as cp

from bats.CudaKernels.load_kernel import load_kernel

KERNEL_FILE = "Backpropagation/compute_factors.cu"
KERNEL_NAME = "compute_factors_kernel"
PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT = 10

__compute_factors_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

#! This causes a problem that is later fed into compute_weights_gradient
#! Problem seems to be that f1 contains nan values
#! fed into this from spike_times that contains nan values
#! x also contains nan values
#! exp_tau contains nan values
#? the question is where does the nan values come from I will now go to LIFFLayerResidual.py to find out, one more method up
def compute_factors(spike_times: cp.array, a: cp.ndarray, c: cp.float32, x: cp.ndarray, exp_tau: cp.ndarray,
                    tau: cp.float32, residual: bool = False) -> Tuple[cp.ndarray, cp.ndarray]:
    
    if residual:
        xfasknfakl=1
    batch_size, n_neurons, max_n_spike = a.shape

    f1 = cp.ndarray(a.shape, dtype=cp.float32)
    f2 = cp.ndarray(a.shape, dtype=cp.float32)
    block_dim = (batch_size, 1, 1)
    grid_dim = (max_n_spike, n_neurons, 2)

    #! Errors found here count: 1 (With BREAKPOINTS)
    __compute_factors_kernel(grid_dim, block_dim, (spike_times, a, c, x, exp_tau, tau,
                                                   f1, f2))
    return f1, f2
