extern "C" {
    __device__ void idx_to_pos(int idx, const int *shape, int *out) {
        int tmp = shape[1] * shape[2];

        out[0] = idx / tmp;
        tmp = idx % tmp;
        out[1] = tmp / shape[2];
        out[2] = tmp % shape[2];
    }

    __global__ void propagate_errors_to_pre_spikes_conv_kernel(const float *f1,
                                                               const float *f2,
                                                               const float *post_times,
                                                               const float *pre_times,
                                                               const float *pre_exp_tau_s,
                                                               const float *pre_exp_tau,
                                                               const float *weights,
                                                               const float *errors,
                                                               const int *post_shape,
                                                               const int *pre_shape,
                                                               const int *filter_shape,
                                                               float *pre_errors,
                                                               int n_post_neuron,
                                                               int max_n_post_spike,
                                                               float tau_s,
                                                               float tau) {
        int sample_idx = threadIdx.x;
        int pre_neuron_idx = blockIdx.x;
        int pre_spike_idx = blockIdx.y;
        int filter_neuron_idx = blockIdx.z;
        int n_filter_neuron = gridDim.z;

        int n_pre_neuron = gridDim.x;
        int max_n_pre_spike = gridDim.y;

        int pre_pos[3];
        int filter_pos[3];
        int filter_shape_2[3] = {filter_shape[1], filter_shape[2], filter_shape[0]};

        idx_to_pos(pre_neuron_idx, pre_shape, pre_pos);
        idx_to_pos(filter_neuron_idx, filter_shape_2, filter_pos);

        int post_pos[3] = {pre_pos[0] - filter_pos[0],
                           pre_pos[1] - filter_pos[1],
                           filter_pos[2]};

        // Check out of input
        if (post_pos[0] < 0 || post_pos[1] < 0 ||
            post_pos[0] >= post_shape[0] || post_pos[1] >= post_shape[1]) {
            return;
        }

        int post_neuron_idx = (post_pos[0] * post_shape[1] + post_pos[1]) * post_shape[2] + post_pos[2];

        unsigned int pre_neuron_offset = sample_idx * n_pre_neuron + pre_neuron_idx;
        unsigned int pre_spike_offset = pre_neuron_offset * max_n_pre_spike + pre_spike_idx;

        pre_times += pre_spike_offset;
        if (isinf(*pre_times)) // No pre spike --> stop
            return;
        pre_exp_tau_s += pre_spike_offset;
        pre_exp_tau += pre_spike_offset;
        pre_errors += pre_spike_offset * n_filter_neuron + filter_neuron_idx;
        //printf("%u %u %u %u\n", pre_spike_offset, n_filter_neuron, filter_neuron_idx,
        //pre_spike_offset * n_filter_neuron + filter_neuron_idx);

        int post_spike_offset = (sample_idx * n_post_neuron + post_neuron_idx) * max_n_post_spike;

        post_times += post_spike_offset;
        f1 += post_spike_offset;
        f2 += post_spike_offset;
        errors += post_spike_offset;
        weights += ((filter_pos[2] * filter_shape[1] + filter_pos[0]) * filter_shape[2] + filter_pos[1])
                    * filter_shape[3] + pre_pos[2];

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