import cupy as cp
import numpy as np

"""
Get indices of spikes sorted in time in an array of spike times per neuron.
"""
def get_sorted_spikes_indices(spike_times_per_neuron, n_spike_per_neuron):
    batch_size, n_neurons, max_n_spike = spike_times_per_neuron.shape
    new_shape = (batch_size, n_neurons * max_n_spike)
    spike_times_reshaped = cp.reshape(spike_times_per_neuron, new_shape)

    total_spikes = cp.sum(n_spike_per_neuron, axis=1)
    max_total_spike = int(cp.max(total_spikes))
    """sorted_indices = cp.argsort(spike_times_reshaped, axis=1)[:, :max_total_spike]"""
    n = np.arange(max_total_spike)
    sorted_indices = cp.argpartition(spike_times_reshaped, n, axis=1)[:, :max_total_spike]
    return new_shape, sorted_indices, spike_times_reshaped