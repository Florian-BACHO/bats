extern "C" {
    __global__ void compute_factors_kernel(const float *spike_times,
                                           const float *a,
                                           const float c,
                                           const float *x,
                                           const float *exp_tau,
                                           float tau,
                                           float *f1,
                                           float *f2) {
        int sample_idx = threadIdx.x;
        int spike_idx = blockIdx.x;
        int neuron_idx = blockIdx.y;
        int factor_idx = blockIdx.z;

        int max_n_spikes = gridDim.x;
        int n_neurons = gridDim.y;
        int neuron_offset = sample_idx * n_neurons + neuron_idx;
        int spike_offset = neuron_offset * max_n_spikes + spike_idx;

        if (isinf(spike_times[spike_offset])) { // No spike, set value to 0.0
            if (factor_idx == 0) // f1
                f1[spike_offset] = 0.0;
            else if (factor_idx == 1) // f2
                f2[spike_offset] = 0.0;
            return;
        }
        if (factor_idx == 0) // f1
            f1[spike_offset] = tau / a[spike_offset] * (1 + exp_tau[spike_offset] * c / x[spike_offset]);
        else // f2
            f2[spike_offset] = tau / x[spike_offset];
    }
}