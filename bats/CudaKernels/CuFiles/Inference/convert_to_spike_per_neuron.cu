extern "C" {
    __global__ void convert_to_spike_per_neuron_kernel(const int *indices,
                                                       const float *times,
                                                       float *spike_per_neuron,
                                                       int *n_per_neuron,
                                                       int n_neurons,
                                                       int max_pre_spike,
                                                       int max_spike_per_neuron) {
        int sample_idx = threadIdx.x;
        int offset = sample_idx * max_pre_spike;

        indices += offset;
        times += offset;
        spike_per_neuron += sample_idx * n_neurons * max_spike_per_neuron;
        n_per_neuron += sample_idx * n_neurons;


        int idx;
        for (int i = 0; i < max_pre_spike; i++) {
            idx = indices[i];
            if (idx < 0)
                break;
            spike_per_neuron[idx * max_spike_per_neuron + n_per_neuron[idx]] = times[i];
            n_per_neuron[idx]++;
        }
    }
}