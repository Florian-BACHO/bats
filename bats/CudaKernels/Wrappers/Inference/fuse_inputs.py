import cupy as cp
import numpy as np

"""
Fuse the inputs of two layers into one input that can be fed to the next layer.
"""
def fuse_inputs(residual_input, jump_input):
    batch_size_res, n_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_neurons_jump, max_n_spike_jump = jump_input.shape
    if batch_size_res != batch_size_jump:
        raise ValueError("Batch size of residual and jump connection must be the same.")
    if max_n_spike_res != max_n_spike_jump:
        max_n_spike = max(max_n_spike_res, max_n_spike_jump)
        residual_input = cp.pad(residual_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_res)),constant_values = cp.inf,mode = 'constant')
        jump_input = cp.pad(jump_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_jump)), constant_values = cp.inf,mode = 'constant')
    
    result = cp.append(residual_input, jump_input, axis=1)
    return result