extern "C" {
    __global__ void get_spike_weights_kernel(const float *weights,
                                             const int *indices,
                                             float *result,
                                             int n_weight_per_neuron) {
        int max_n_spike = gridDim.x;
        int n_neurons = gridDim.y;

        int sample_idx = threadIdx.x;
        int pre_spike_idx = blockIdx.x;
        int neuron_idx = blockIdx.y;

        int spike_idx = (sample_idx * max_n_spike) + pre_spike_idx;

        int pre_neuron_idx = indices[spike_idx];
        int weight_idx = (n_weight_per_neuron * neuron_idx) + pre_neuron_idx;
        int result_idx = (sample_idx * n_neurons + neuron_idx) * max_n_spike + pre_spike_idx;

        result[result_idx] = weights[weight_idx];
    }
}