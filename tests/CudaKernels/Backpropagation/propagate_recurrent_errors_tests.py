import unittest
import numpy as np
import cupy as cp

from bats.CudaKernels.Wrappers.Backpropagation import propagate_recurrent_errors


class CudaKernelsTests(unittest.TestCase):
    def prop_recurrent_errors(self, x, exp_tau, errors, delta_theta_tau):
        for sample_x, sample_exp_tau, sample_errors in zip(x, exp_tau, errors):
            for neuron_x, neuron_exp_tau, neuron_errors in zip(sample_x, sample_exp_tau, sample_errors):
                for i in reversed(range(len(neuron_errors))):
                    if neuron_errors[i] == 0.0:
                        continue
                    for j in range(i + 1, len(neuron_errors)):
                        if neuron_errors[j] == 0.0:
                            break
                        neuron_errors[i] += neuron_errors[j] * delta_theta_tau / neuron_x[j] * neuron_exp_tau[i]

    def test_propagate_recurrent_errors(self):
        delta_theta = 1.0
        tau = 0.2
        delta_theta_tau = delta_theta / tau
        spikes = np.array([[[0.1, 0.2, 0.3, 0.4],
                            [0.3, 0.35, 0.4, np.inf],
                            [np.inf, np.inf, np.inf, np.inf],
                            [0.4, 1.0, np.inf, np.inf]],
                           [[0.2, 0.3, 0.35, 0.5],
                            [0.2, 0.4, 0.7, np.inf],
                            [np.inf, np.inf, np.inf, np.inf],
                            [0.7, np.inf, np.inf, np.inf]]])
        x = spikes
        exp_tau = np.exp(spikes / tau)
        errors = np.array([[[1.0, 2.0, -3.0, 4.0],
                            [3.0, 3.5, -4.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.4, -1.0, 0.0, 0.0]],
                           [[-0.2, -0.3, -0.35, 0.5],
                            [-1.2, 4.0, -0.7, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [4.2, 0.0, 0.0, 0.0]]])

        x_gpu = cp.array(x, dtype=cp.float32)
        exp_tau_gpu = cp.array(exp_tau, dtype=cp.float32)
        errors_gpu = cp.array(errors, dtype=cp.float32)

        self.prop_recurrent_errors(x, exp_tau, errors, delta_theta_tau)
        propagate_recurrent_errors(x_gpu, exp_tau_gpu, errors_gpu, cp.float32(delta_theta_tau))

        self.assertTrue(np.allclose(errors, errors_gpu.get()))