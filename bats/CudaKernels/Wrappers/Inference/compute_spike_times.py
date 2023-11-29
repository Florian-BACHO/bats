from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp
import numpy as np

KERNEL_FILE = "Inference/compute_spike_times.cu"
KERNEL_NAME = "compute_spike_times_kernel"

__compute_spike_times_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

#! WE FOUND THE PROBLEM
#! IT is here somewhere as this method creates the nan values

# ALL shapedlike (50,230) and spike_weights shaped like (50,64,230) for both residual and non residual
def compute_spike_times(spike_times: cp.ndarray,
                        exp_tau_s: cp.ndarray, exp_tau: cp.ndarray,
                        spike_weights: cp.ndarray,
                        c: cp.float32, delta_theta_tau: np.float32, tau: np.float32,
                        max_simulation: np.float32, max_n_post_spikes: np.int32, residual: bool = False):

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
    
    #! this works wrong it adds NaN
    # if residual:
    #     x= 1
    # else:
    #     x = 0
    __compute_spike_times_kernel(grid_dim, block_dim, args)

    # if np.isnan(post_spike_times).any():
    #     # Some spikes are not computed
    #     # I just could replace the nan values with the max_simulation value
    #     # But I am not sure if this is the right thing to do
    #     # I think the problem is in the kernel
    #     # I am going to do it anyway
    #     #TODO: check if this is the right thing to do and improve on it -> APPARENTLY NOT NEEDED XD
    #     post_spike_times= cp.nan_to_num(post_spike_times, nan=cp.inf, posinf=cp.inf)
        
    # if np.isnan(x).any():
    #     x= cp.nan_to_num(x, nan=cp.inf, posinf=cp.inf)
    
    # if np.isnan(a).any():
    #     a= cp.nan_to_num(a, nan=cp.inf, posinf=cp.inf)
    
    # if np.isnan(post_exp_tau).any():
    #     post_exp_tau= cp.nan_to_num(post_exp_tau, nan=cp.inf, posinf=cp.inf)

    return n_spikes, a, x, post_spike_times, post_exp_tau
