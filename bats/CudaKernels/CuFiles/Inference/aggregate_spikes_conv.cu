extern "C" {
    __device__ int compute_idx(int sample_idx, const int *pos, const int *shape) {
        return ((sample_idx * shape[0] + pos[0]) * shape[1] + pos[1]) * shape[2] + pos[2];
    }

    __global__ void aggregate_spikes_conv_kernel(const int *n_spikes,
                                                 const float *spike_times,
                                                 const int *pre_shape,
                                                 int max_pre_spike,
                                                 int *out_n_spikes,
                                                 float *out_spike_times,
                                                 int *out_spike_indices) {
        int sample_idx = threadIdx.x;
        int post_pos[3] = {blockIdx.x, blockIdx.y, blockIdx.z};
        int post_shape[3] = {gridDim.x, gridDim.y, gridDim.z};
        int pre_pos[3] = {post_pos[0] * 2,
                          post_pos[1] * 2,
                          post_pos[2]};

        int tmp_idx = compute_idx(sample_idx, post_pos, post_shape);
        out_n_spikes += tmp_idx;
        tmp_idx = tmp_idx * 4 * max_pre_spike;
        out_spike_times += tmp_idx;
        out_spike_indices += tmp_idx;

        for (int x=0; x < 2; x++) {
            pre_pos[0] = post_pos[0] * 2 + x;
            for (int y=0; y < 2; y++) {
                pre_pos[1] = post_pos[1] * 2 + y;
                tmp_idx = compute_idx(sample_idx, pre_pos, pre_shape);

                (*out_n_spikes) += n_spikes[tmp_idx];

                tmp_idx *= max_pre_spike;
                for (int i=0; i < max_pre_spike; i++) {
                    *out_spike_times = spike_times[tmp_idx + i];
                    *out_spike_indices = tmp_idx + i;
                    out_spike_times++;
                    out_spike_indices++;
                }
            }
        }
    }
}