extern "C" {
    __global__ void propagate_recurrent_errors_kernel(const float *x,
                                                      const float *exp_tau,
                                                      float *errors,
                                                      float delta_theta_tau,
                                                      int max_n_spike) {
        int sample_idx = threadIdx.x;
        int neuron_idx = blockIdx.x;
        int n_neurons = gridDim.x;
        int offset = (sample_idx * n_neurons + neuron_idx) * max_n_spike;
        float cumulated_error = 0.0;

        x += offset;
        exp_tau += offset;
        errors += offset;

        for (int i = max_n_spike - 1; i >= 0; i--) {
            if (isinf(exp_tau[i]))
                continue;
            errors[i] += cumulated_error * exp_tau[i];
            cumulated_error += errors[i] * delta_theta_tau / x[i];
        }
    }
}