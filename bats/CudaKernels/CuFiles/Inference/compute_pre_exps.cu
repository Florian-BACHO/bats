extern "C" {
    __global__ void compute_pre_exps_kernel(const float *values,
                                            float *exp_tau_s,
                                            float *exp_tau,
                                            float tau_s,
                                            float tau) {
        int max_n_spike = gridDim.x;
        int n_neurons = gridDim.y;

        int sample_idx = threadIdx.x;
        int pre_spike_idx = blockIdx.x;
        int neuron_idx = blockIdx.y;
        int tau_idx = blockIdx.z; // 0 if tau_s, 1 if tau

        // Global pre-spike index (i.e. relative to all samples' spikes)
        int spike_idx = ((sample_idx * n_neurons + neuron_idx) * max_n_spike) + pre_spike_idx;

        if (tau_idx == 0)
            exp_tau_s[spike_idx] = __expf(values[spike_idx] / tau_s);
        else
            exp_tau[spike_idx] = __expf(values[spike_idx] / tau);
    }
}