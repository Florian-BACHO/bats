extern "C" {
    __global__ void apply_label_weights_synaptic_scaling_kernel(const float *weights,
                                                                const float *labels,
                                                                const int *n_spike_per_neuron,
                                                                float *gradient,
                                                                float factor,
                                                                int n_post) {
        int sample_idx = threadIdx.x;
        int label = labels[sample_idx];

        if (n_spike_per_neuron[label] > 0) // Label neuron spiked --> stop
            return;

        int pre_idx = blockIdx.x;
        int n_pre = gridDim.y;

        gradient[(sample_idx * n_post + label) * n_pre + pre_idx] -=
            factor * fabsf(weights[label * n_pre + pre_idx]);
    }
}