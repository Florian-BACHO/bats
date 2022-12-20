extern "C" {
    __global__ void compute_weights_gradient_kernel(const float *f1,
                                                    const float *f2,
                                                    const float *post_times,
                                                    const float *pre_times,
                                                    const float *pre_exp_tau_s,
                                                    const float *pre_exp_tau,
                                                    const float *errors,
                                                    float *spikes_gradient,
                                                    int max_n_pre_spike) {
        int sample_idx = threadIdx.x;
        int post_neuron_idx = blockIdx.x;
        int post_spike_idx = blockIdx.y;
        int pre_neuron_idx = blockIdx.z;

        int n_post_neuron = gridDim.x;
        int max_n_post_spike = gridDim.y;
        int n_pre_neuron = gridDim.z;

        int post_neuron_offset = sample_idx * n_post_neuron + post_neuron_idx;
        int post_spike_offset = post_neuron_offset * max_n_post_spike + post_spike_idx;

        post_times += post_spike_offset;
        if (isinf(*post_times)) // No spike or no error --> stop
            return;

        f1 += post_spike_offset;
        f2 += post_spike_offset;
        errors += post_spike_offset;
        spikes_gradient += post_spike_offset * n_pre_neuron + pre_neuron_idx;

        int pre_offset = (sample_idx * n_pre_neuron + pre_neuron_idx) * max_n_pre_spike;
        pre_exp_tau_s += pre_offset;
        pre_exp_tau += pre_offset;
        pre_times += pre_offset;

        float sum_exp_tau_s = 0.0;
        float sum_exp_tau = 0.0;
        for (int i = 0; i < max_n_pre_spike; i++) {
            // No pre-spike for the pre-synaptic neuron anymore or pre-spike occurs after the post spike --> stop
            if (isinf(pre_times[i]) || pre_times[i] >= *post_times)
                break;
            sum_exp_tau_s += pre_exp_tau_s[i];
            sum_exp_tau += pre_exp_tau[i];
        }
        *spikes_gradient = -*errors * (*f1 * sum_exp_tau_s + *f2 * sum_exp_tau);
    }
}