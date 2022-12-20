extern "C" {
    __device__ void idx_to_pos(int idx, const int *shape, int *out) {
        int tmp = shape[1] * shape[2];

        out[0] = idx / tmp;
        tmp = idx % tmp;
        out[1] = tmp / shape[2];
        out[2] = tmp % shape[2];
    }

    __global__ void compute_weights_gradient_conv_kernel(const float *f1,
                                                         const float *f2,
                                                         const float *post_times,
                                                         const float *pre_times,
                                                         const float *pre_exp_tau_s,
                                                         const float *pre_exp_tau,
                                                         const float *errors,
                                                         const int *pre_shape,
                                                         const int *post_shape,
                                                         const int *filter_shape,
                                                         float *gradient,
                                                         int n_post_neuron,
                                                         int n_pre_neuron,
                                                         int max_n_post_spike,
                                                         int max_n_pre_spike) {
        int sample_idx = threadIdx.x;
        int filter_pos[3] = {blockIdx.x, blockIdx.y, blockIdx.z};
        int post_neuron_pos[3];
        int pre_pos[3];

        for (int post_neuron_idx = 0; post_neuron_idx < n_post_neuron; post_neuron_idx++) {
            idx_to_pos(post_neuron_idx, post_shape, post_neuron_pos);

            pre_pos[0] = post_neuron_pos[0] + filter_pos[0];
            pre_pos[1] = post_neuron_pos[1] + filter_pos[1];
            pre_pos[2] = filter_pos[2];
            int pre_neuron_idx = (pre_pos[0] * pre_shape[1] + pre_pos[1]) * pre_shape[2] + pre_pos[2];

            int post_neuron_offset = sample_idx * n_post_neuron + post_neuron_idx;
            int post_spike_offset = post_neuron_offset * max_n_post_spike;

            int gradient_idx = (((sample_idx * filter_shape[0] + post_neuron_pos[2]) * filter_shape[1] + filter_pos[0])
                                * filter_shape[2] + filter_pos[1]) * filter_shape[3] + filter_pos[2];

            int pre_offset = (sample_idx * n_pre_neuron + pre_neuron_idx) * max_n_pre_spike;

            float sum_exp_tau_s = 0.0;
            float sum_exp_tau = 0.0;
            int j = pre_offset;
            for (int i = post_spike_offset; i < (post_spike_offset + max_n_post_spike); i++) {
                if (isinf(post_times[i])) // No spike or no error --> stop
                    break;
                while (j < (pre_offset + max_n_pre_spike) && pre_times[j] < post_times[i]) {
                    sum_exp_tau_s += pre_exp_tau_s[j];
                    sum_exp_tau += pre_exp_tau[j];
                    j++;
                }
                gradient[gradient_idx] += errors[i] * (f1[i] * sum_exp_tau_s - f2[i] * sum_exp_tau);
            }
        }
    }
}