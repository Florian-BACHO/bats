#define INFINITY __int_as_float(0x7f800000)

extern "C" {
    __device__ void get_sample_params(const int **spike_indices,
                                      const float **spike_times,
                                      const float **exp_tau_s,
                                      const float **exp_tau,
                                      const float **weights,
                                      int n_neurons,
                                      int sample_idx,
                                      int max_n_pre_spike,
                                      int neuron_c_idx,
                                      int n_weight_per_filter) {
        int sample_start_idx = sample_idx * max_n_pre_spike;


        *spike_indices += sample_start_idx;
        *spike_times += sample_start_idx;
        *exp_tau_s += sample_start_idx;
        *exp_tau += sample_start_idx;
        *weights += neuron_c_idx * n_weight_per_filter;
    }

    __device__ void get_neuron_results(int **n_spikes,
                                       float **a,
                                       float **x,
                                       float **spike_times,
                                       float **post_exp_tau,
                                       int n_neurons, int sample_idx, int neuron_idx, int max_n_post_spike) {
        int sample_neuron_idx = (sample_idx * n_neurons + neuron_idx);
        int res_start_idx = sample_neuron_idx * max_n_post_spike;

        *n_spikes += sample_neuron_idx;
        *a += res_start_idx;
        *x += res_start_idx;
        *spike_times += res_start_idx;
        *post_exp_tau += res_start_idx;
    }

    __device__ bool compute_spikes(const float c,
                                   int *n_spikes,
                                   float *a,
                                   float *x,
                                   float *spike_times,
                                   float *post_exp_tau,
                                   float cumul_a,
                                   float *cumul_b,
                                   float last_spike,
                                   float next_spike,
                                   float delta_theta_tau,
                                   float tau,
                                   float max_simulation,
                                   int neuron_idx,
                                   int max_n_post_spike,
                                   int sample_idx) {
        float x_tmp, inside_log, tmp;

        // Compute until there is no spike anymore
        while (true) {
            tmp = (*cumul_b) * (*cumul_b) - 4 * cumul_a * c;

            if (tmp < 0) // Negative square root, no spike --> stop
                return false;
            x_tmp = sqrtf(tmp);
            tmp = x_tmp + (*cumul_b);

            if (tmp == 0.0) // Division per zero, no spike --> stop
                return false;
            inside_log = 2 * cumul_a / tmp;
            if (inside_log < 0) // Negative log, no spike --> stop
                return false;

            tmp = tau * __logf(inside_log);
            // Spike time is before the last pre-spike or after the next spike --> stop
            if (tmp <= last_spike || tmp > max_simulation || tmp > next_spike) {
                return false;
            }

            // Spike time is valid
            a[*n_spikes] = cumul_a;
            x[*n_spikes] = x_tmp;
            spike_times[*n_spikes] = tmp;
            last_spike = tmp;
            post_exp_tau[*n_spikes] = inside_log;
            *cumul_b -= delta_theta_tau * inside_log; // Apply reset to b
            (*n_spikes)++;
            if (*n_spikes >= max_n_post_spike) {
                return true;
            }
        }
    }

   __device__ float get_spike_weight(const float *weights,
                                     const int *pre_shape,
                                     const int *neuron_idx_3d,
                                     const int *filters_shape,
                                     const int *n_neurons_3d,
                                     int spike_idx) {
        int tmp = pre_shape[1] * pre_shape[2];
        int spike_x = spike_idx / tmp;
        int spike_w = spike_idx % tmp;
        int spike_y = spike_w / pre_shape[2];
        int spike_c = spike_w % pre_shape[2];

        if (spike_x < neuron_idx_3d[0] || spike_x >= (neuron_idx_3d[0] + filters_shape[1]) ||
            spike_y < neuron_idx_3d[1] || spike_y >= (neuron_idx_3d[1] + filters_shape[2])) {
            return 0.0;
        }

        int pos_x = spike_x - neuron_idx_3d[0];
        int pos_y = spike_y - neuron_idx_3d[1];
        int weight_idx = (pos_x * filters_shape[2] + pos_y) * filters_shape[3] + spike_c;
        //printf("%d %d | %d %d | %f\n", neuron_idx_3d[0], neuron_idx_3d[1], pos_x, pos_y, weights[weight_idx]);
        return weights[weight_idx];
   }

    __global__ void compute_spike_times_conv_kernel(// Parameters
                                               const int *spike_indices,
                                               const float *spike_times,
                                               const float *exp_tau_s,
                                               const float *exp_tau,
                                               const float *weights,
                                               const int *pre_shape,
                                               const int *post_shape,
                                               const int *filters_shape,
                                               int n_neurons,
                                               const float c,
                                               float delta_theta_tau,
                                               float tau,
                                               float max_simulation,
                                               int max_n_pre_spike,
                                               int max_n_post_spike,
                                               // Outputs
                                               int *n_spikes,
                                               float *a,
                                               float *x,
                                               float *out_spike_times,
                                               float *post_exp_tau) {
        int sample_idx = threadIdx.x;
        int neuron_idx_3d[3] = {blockIdx.x, blockIdx.y, blockIdx.z};
        int neuron_idx = (blockIdx.x * post_shape[1] + blockIdx.y) * post_shape[2] + blockIdx.z;
        int n_weight_per_filter = filters_shape[1] * filters_shape[2] * filters_shape[3];

        get_sample_params(&spike_indices, &spike_times, &exp_tau_s, &exp_tau, &weights,
                          n_neurons, sample_idx, max_n_pre_spike, neuron_idx_3d[2], n_weight_per_filter);
        get_neuron_results(&n_spikes, &a, &x, &out_spike_times, &post_exp_tau,
                           n_neurons, sample_idx, neuron_idx, max_n_post_spike);

        float cumul_a = 0.0;
        float cumul_b = 0.0;
        float weight;
        int next_i;
        float next_spike;

        for (int i = 0; i < max_n_pre_spike; i++) {
            if (spike_times[i] == INFINITY) // No spike anymore --> stop
                break;
            weight = get_spike_weight(weights, pre_shape, neuron_idx_3d, filters_shape, post_shape, spike_indices[i]);

            cumul_a += weight * exp_tau_s[i];
            cumul_b += weight * exp_tau[i];

            next_i = i + 1;
            if (next_i < max_n_pre_spike)
                next_spike = spike_times[next_i];
            else
                next_spike = INFINITY;

            if (compute_spikes(c, n_spikes, a, x, out_spike_times, post_exp_tau,
                               cumul_a, &cumul_b, spike_times[i], next_spike, delta_theta_tau, tau,
                               max_simulation, neuron_idx, max_n_post_spike, sample_idx))
                break; // Buffer full
        }
    }
}