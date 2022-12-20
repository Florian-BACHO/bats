import unittest
import numpy as np
import cupy as cp
from bats.Losses import TTFSSoftmaxCrossEntropy


class TTFSSoftmaxCrossEntropy_tests(unittest.TestCase):
    def compute_loss_cpu(self, spikes_per_neuron, labels, tau):
        first_spikes = spikes_per_neuron[..., 0]
        exps = np.exp(-first_spikes / tau)
        sums = np.sum(exps, axis=1)
        labels_exps = exps[np.arange(len(labels)), labels]
        softmax = labels_exps / sums
        return -np.log(softmax)

    def test_compute_loss(self):
        tau = 0.2
        spike_per_neuron = np.array([[[0.1, 0.2, 0.3, np.inf],
                                      [0.2, np.inf, np.inf, np.inf],
                                      [np.inf, np.inf, np.inf, np.inf],
                                      [0.3, 0.4, 0.5, 0.6]],
                                     [[0.4, 0.5, np.inf, np.inf],
                                      [1.0, np.inf, np.inf, np.inf],
                                      [0.9, 1.1, np.inf, np.inf],
                                      [0.5, np.inf, np.inf, np.inf]]])
        spike_per_neuron_gpu = cp.array(spike_per_neuron, dtype=cp.float32)

        labels = np.array([3, 0])
        labels_gpu = cp.array(labels)

        loss_fct = TTFSSoftmaxCrossEntropy(tau)
        loss = loss_fct.compute_loss(spike_per_neuron_gpu, cp.empty((0,)), labels_gpu)

        true_loss = self.compute_loss_cpu(spike_per_neuron, labels, tau)

        self.assertTrue(np.allclose(loss.get(), true_loss))

    def compute_errors_cpu(self, spikes_per_neuron, labels, tau):
        first_spikes = spikes_per_neuron[..., 0]
        exps = np.exp(-first_spikes / tau)
        sums = np.sum(exps, axis=1)
        neg_softmax = -exps / sums[:, np.newaxis]
        neg_softmax[np.arange(labels.size), labels] += 1
        neg_softmax /= tau
        errors = np.zeros(spikes_per_neuron.shape, dtype=cp.float32)
        errors[..., 0] = neg_softmax
        return errors

    def test_compute_errors(self):
        tau = 0.2
        spike_per_neuron = np.array([[[0.1, 0.2, 0.3, np.inf],
                                      [0.2, np.inf, np.inf, np.inf],
                                      [np.inf, np.inf, np.inf, np.inf],
                                      [0.3, 0.4, 0.5, 0.6]],
                                     [[0.4, 0.5, np.inf, np.inf],
                                      [1.0, np.inf, np.inf, np.inf],
                                      [0.9, 1.1, np.inf, np.inf],
                                      [0.5, np.inf, np.inf, np.inf]]])
        spike_per_neuron_gpu = cp.array(spike_per_neuron, dtype=cp.float32)

        labels = np.array([3, 0])
        labels_gpu = cp.array(labels)

        loss_fct = TTFSSoftmaxCrossEntropy(tau)
        errors = loss_fct.compute_errors(spike_per_neuron_gpu, cp.empty((0,)), labels_gpu)
        true_errors = self.compute_errors_cpu(spike_per_neuron, labels, tau)

        self.assertTrue(np.allclose(errors.get(), true_errors))

    def test_compute_loss_and_errors(self):
        tau = 0.2
        spike_per_neuron = np.array([[[0.1, 0.2, 0.3, np.inf],
                                      [0.2, np.inf, np.inf, np.inf],
                                      [np.inf, np.inf, np.inf, np.inf],
                                      [0.3, 0.4, 0.5, 0.6]],
                                     [[0.4, 0.5, np.inf, np.inf],
                                      [1.0, np.inf, np.inf, np.inf],
                                      [0.9, 1.1, np.inf, np.inf],
                                      [0.5, np.inf, np.inf, np.inf]]])
        spike_per_neuron_gpu = cp.array(spike_per_neuron, dtype=cp.float32)

        labels = np.array([3, 0])
        labels_gpu = cp.array(labels)

        loss_fct = TTFSSoftmaxCrossEntropy(tau)
        loss, errors = loss_fct.compute_loss_and_errors(spike_per_neuron_gpu, cp.empty((0,)), labels_gpu)

        true_errors = self.compute_errors_cpu(spike_per_neuron, labels, tau)
        true_loss = self.compute_loss_cpu(spike_per_neuron, labels, tau)

        self.assertTrue(np.allclose(loss.get(), true_loss))
        self.assertTrue(np.allclose(errors.get(), true_errors))