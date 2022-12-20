import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import compute_weights_gradient, compute_weights_gradient_conv
from bats.CudaKernels.Wrappers.Inference import convert_to_spike_per_neuron, aggregate_spikes_conv


class CudaKernelsTests(unittest.TestCase):
    def idx_to_pos(self, idx, shape):
        tmp = shape[1] * shape[2]
        x = idx / tmp
        tmp = idx % tmp
        y = tmp / shape[2]
        z = tmp % shape[2]
        return int(x), int(y), int(z)

    def aggregate_spikes_conv_cpu(self, n_spikes_per_neuron, spike_times, pre_shape, post_shape):
        batch_size, _, max_pre_spikes = spike_times.shape
        pre_x, pre_y, pre_c = pre_shape
        post_x, post_y, post_c = post_shape
        n_post = np.prod(post_shape)
        n_spikes_reshaped = n_spikes_per_neuron.reshape((batch_size, pre_x, pre_y, pre_c))
        spike_times_reshaped = spike_times.reshape((batch_size, pre_x, pre_y, pre_c, max_pre_spikes))
        out_n_spikes = np.zeros((batch_size, post_x, post_y, post_c), dtype=np.int32)
        out_spike_times = np.empty((batch_size, post_x, post_y, post_c, max_pre_spikes * 4), dtype=np.float32)
        out_spike_indices = np.empty((batch_size, post_x, post_y, post_c, max_pre_spikes * 4), dtype=np.int32)
        indices = np.arange(0, spike_times.size).reshape(spike_times_reshaped.shape)
        for sample in range(batch_size):
            for x in range(post_x):
                for y in range(post_y):
                    x_pre = x * 2
                    y_pre = y * 2
                    tmp = n_spikes_reshaped[:, x_pre:x_pre + 2, y_pre:y_pre + 2, :].reshape((batch_size, 4, pre_c))
                    out_n_spikes[:, x, y, :] = np.sum(tmp, axis=1)

                    out_spike_times[:, x, y, :, 0:max_pre_spikes] = spike_times_reshaped[:, x_pre, y_pre, :, :]
                    out_spike_times[:, x, y, :, max_pre_spikes:2 * max_pre_spikes] = \
                        spike_times_reshaped[:, x_pre, y_pre + 1, :, :]
                    out_spike_times[:, x, y, :, 2 * max_pre_spikes:3 * max_pre_spikes] = \
                        spike_times_reshaped[:, x_pre + 1, y_pre, :, :]
                    out_spike_times[:, x, y, :, 3 * max_pre_spikes:4 * max_pre_spikes] = \
                        spike_times_reshaped[:, x_pre + 1, y_pre + 1, :, :]

                    out_spike_indices[:, x, y, :, 0:max_pre_spikes] = indices[:, x_pre, y_pre, :, :]
                    out_spike_indices[:, x, y, :, max_pre_spikes:2 * max_pre_spikes] = \
                        indices[:, x_pre, y_pre + 1, :, :]
                    out_spike_indices[:, x, y, :, 2 * max_pre_spikes:3 * max_pre_spikes] = \
                        indices[:, x_pre + 1, y_pre, :, :]
                    out_spike_indices[:, x, y, :, 3 * max_pre_spikes:4 * max_pre_spikes] = \
                        indices[:, x_pre + 1, y_pre + 1, :, :]

        out_n_spikes = out_n_spikes.reshape((batch_size, n_post))

        out_spike_times = out_spike_times.reshape(batch_size, n_post, 4 * max_pre_spikes)
        out_spike_indices = out_spike_indices.reshape(batch_size, n_post, 4 * max_pre_spikes)
        sorted_indices = np.argsort(out_spike_times, axis=2)
        out_spike_times = np.take_along_axis(out_spike_times, sorted_indices, axis=2)
        out_spike_indices = np.take_along_axis(out_spike_indices, sorted_indices, axis=2)

        return out_n_spikes, out_spike_times, out_spike_indices

    def test_aggregate_spikes_conv(self):
        batch_size = 1
        pre_shape = np.array([10, 10, 3])
        pre_shape_gpu = cp.array(pre_shape, dtype=cp.int32)
        post_shape = np.array([pre_shape[0] // 2, pre_shape[1] // 2, pre_shape[2]])
        post_shape_gpu = cp.array(post_shape, dtype=cp.int32)
        n_pre = np.prod(pre_shape)
        n_post = np.prod(post_shape)

        nb_spikes = 100

        spike_indices_cpu = np.random.randint(0, n_pre, size=(batch_size, nb_spikes))
        spike_indices_gpu = cp.array(spike_indices_cpu.astype(np.int32))

        spike_times_cpu = np.sort(np.random.uniform(0, 1.0, size=(batch_size, nb_spikes)), axis=1)
        spike_times_cpu[0, -30:] = np.inf
        spike_times_gpu = cp.array(spike_times_cpu.astype(np.float32))

        spike_times, n_spikes_per_neuron = convert_to_spike_per_neuron(spike_indices_gpu, spike_times_gpu,
                                                                       cp.int32(n_pre))

        post_n_spikes, post_spike_times, post_spike_indices = aggregate_spikes_conv(n_spikes_per_neuron, spike_times,
                                                                                    pre_shape_gpu, post_shape_gpu)

        true_post_n_spikes, true_post_spike_times, true_post_spike_indices = \
            self.aggregate_spikes_conv_cpu(n_spikes_per_neuron.get(), spike_times.get(), pre_shape, post_shape)

        np.testing.assert_allclose(post_n_spikes.get(), true_post_n_spikes)
        np.testing.assert_allclose(post_spike_times.get(), true_post_spike_times, atol=1e-3, rtol=0)
        np.testing.assert_allclose(post_spike_indices.get(), true_post_spike_indices)
