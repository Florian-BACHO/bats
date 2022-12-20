extern "C" {
    __global__ void compute_weights_gradient_kernel(const float *f1,
                                                    const float *f2,
                                                    const float *post_times,
                                                    const float *pre_times,
                                                    const float *pre_exp_tau_s,
                                                    const float *pre_exp_tau,
                                                    const float *errors,
                                                    float *gradient,
                                                    int max_n_post_spike,
                                                    int max_n_pre_spike) {
        int sample_idx = threadIdx.x;
        int post_neuron_idx = blockIdx.x;
        int pre_neuron_idx = blockIdx.y;

        int n_post_neuron = gridDim.x;
        int n_pre_neuron = gridDim.y;

        int post_neuron_offset = sample_idx * n_post_neuron + post_neuron_idx;
        int post_spike_offset = post_neuron_offset * max_n_post_spike;

        post_times += post_spike_offset;
        f1 += post_spike_offset;
        f2 += post_spike_offset;
        errors += post_spike_offset;
        gradient += post_neuron_offset * n_pre_neuron + pre_neuron_idx;

        int pre_offset = (sample_idx * n_pre_neuron + pre_neuron_idx) * max_n_pre_spike;
        pre_exp_tau_s += pre_offset;
        pre_exp_tau += pre_offset;
        pre_times += pre_offset;

        float sum_exp_tau_s = 0.0;
        float sum_exp_tau = 0.0;
        int j = 0;
        for (int i = 0; i < max_n_post_spike; i++) {
            if (isinf(post_times[i])) // No spike or no error --> stop
                break;
            while (j < max_n_pre_spike && pre_times[j] < post_times[i]) {
                sum_exp_tau_s += pre_exp_tau_s[j];
                sum_exp_tau += pre_exp_tau[j];
                j++;
            }
            *gradient += errors[i] * (f1[i] * sum_exp_tau_s - f2[i] * sum_exp_tau);
        }
    }
}