extern "C" {
    __global__ void apply_weights_synaptic_scaling_kernel(const float *weights,
                                                          const float *silent_ratios,
                                                          const int *n_spike_per_neuron,
                                                          float *gradient,
                                                          float scaling_threshold,
                                                          float factor) {
        int sample_idx = threadIdx.x;


        if (silent_ratios[sample_idx] < scaling_threshold) // Silent ratio below threshold --> stop
            return;

        int post_idx = blockIdx.x;
        int pre_idx = blockIdx.y;

        int n_post = gridDim.x;
        int n_pre = gridDim.y;

        int tmp = sample_idx * n_post + post_idx;
        if (n_spike_per_neuron[tmp] > 0) // Only scale silent neurons
            return;

        gradient[tmp * n_pre + pre_idx] -= factor * fabsf(weights[post_idx * n_pre + pre_idx]);
    }
}