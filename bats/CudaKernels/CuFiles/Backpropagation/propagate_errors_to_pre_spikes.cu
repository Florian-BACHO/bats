extern "C" {
    __global__ void propagate_errors_to_pre_spikes_kernel(const float *f1,
                                                          const float *f2,
                                                          const float *post_times,
                                                          const float *pre_times,
                                                          const float *pre_exp_tau_s,
                                                          const float *pre_exp_tau,
                                                          const float *weights,
                                                          const float *errors,
                                                          float *pre_errors,
                                                          int max_n_post_spike,
                                                          float tau_s,
                                                          float tau) {
        int sample_idx = threadIdx.x;
        int pre_neuron_idx = blockIdx.x;
        int pre_spike_idx = blockIdx.y;
        int post_neuron_idx = blockIdx.z;

        int n_pre_neuron = gridDim.x;
        int max_n_pre_spike = gridDim.y;
        int n_post_neuron = gridDim.z;

        int pre_neuron_offset = sample_idx * n_pre_neuron + pre_neuron_idx;
        int pre_spike_offset = pre_neuron_offset * max_n_pre_spike + pre_spike_idx;

        pre_times += pre_spike_offset;
        if (isinf(*pre_times)) // No pre spike --> stop
            return;
        pre_exp_tau_s += pre_spike_offset;
        pre_exp_tau += pre_spike_offset;
        pre_errors += pre_spike_offset * n_post_neuron + post_neuron_idx;

        int post_spike_offset = (sample_idx * n_post_neuron + post_neuron_idx) * max_n_post_spike;

        post_times += post_spike_offset;
        f1 += post_spike_offset;
        f2 += post_spike_offset;
        errors += post_spike_offset;
        weights += post_neuron_idx * n_pre_neuron + pre_neuron_idx;

        float cumul_f1 = 0.0;
        float cumul_f2 = 0.0;
        for (int i = 0; i < max_n_post_spike; i++) {
            // No post-spike --> stop
            if (isinf(post_times[i]))
                break;
            // Post synaptic spike is before the pre-synaptic spike: no error to propagate --> skip
            if (post_times[i] <= *pre_times)
                continue;
            cumul_f1 += errors[i] * f1[i];
            cumul_f2 += errors[i] * f2[i];
        }
        *pre_errors += *weights * (cumul_f1 * *pre_exp_tau_s / tau_s - cumul_f2 * *pre_exp_tau / tau);
    }
}