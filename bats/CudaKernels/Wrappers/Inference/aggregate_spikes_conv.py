from typing import Tuple
from bats.CudaKernels.load_kernel import load_kernel
import cupy as cp

KERNEL_FILE = "Inference/aggregate_spikes_conv.cu"
KERNEL_NAME = "aggregate_spikes_conv_kernel"

__aggregate_spikes_conv_kernel = load_kernel(KERNEL_FILE, KERNEL_NAME)

def aggregate_spikes_conv(n_spikes: cp.ndarray, spike_times: cp.ndarray,
                          pre_shape: cp.ndarray, post_shape: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    batch_size, n_neurons, max_pre_spike = spike_times.shape
    post_x, post_y, post_c = post_shape.get()

    n_post = post_x * post_y * post_c
    out_n_spikes = cp.zeros((batch_size, n_post), dtype=cp.int32)
    out_spike_times = cp.empty((batch_size, n_post, max_pre_spike * 4), dtype=cp.float32)
    out_spike_indices = cp.empty((batch_size, n_post, max_pre_spike * 4), dtype=cp.int32)

    block_dim = (batch_size, 1, 1)
    grid_dim = (post_x, post_y, post_c)
    __aggregate_spikes_conv_kernel(grid_dim, block_dim, (n_spikes, spike_times, pre_shape, max_pre_spike,
                                                         out_n_spikes, out_spike_times, out_spike_indices))
    sorted_indices = cp.argsort(out_spike_times, axis=2)
    out_spike_times = cp.take_along_axis(out_spike_times, sorted_indices, axis=2)
    out_spike_indices = cp.take_along_axis(out_spike_indices, sorted_indices, axis=2)

    return out_n_spikes, out_spike_times, out_spike_indices