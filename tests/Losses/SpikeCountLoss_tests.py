import unittest
import numpy as np
import cupy as cp
from bats.Losses import SpikeCountClassLoss


class SpikeCOuntLoss_tests(unittest.TestCase):
    def test_compute_loss(self):
        batch_size = 100
        n_neuron = 10
        max_spike = 15
        n_spike_per_neuron = np.random.randint(0, max_spike, size=(batch_size, n_neuron))
        n_spike_per_neuron_gpu = cp.array(n_spike_per_neuron, dtype=cp.float32)
        target_true = 10
        target_false = 2

        labels = np.random.randint(0, n_neuron, size=(batch_size,))
        labels_gpu = cp.array(labels, dtype=cp.int32)

        loss_fct = SpikeCountClassLoss(target_true=target_true, target_false=target_false)
        loss = loss_fct.compute_loss(cp.empty((0,)), n_spike_per_neuron_gpu, labels_gpu)

        targets = np.full((batch_size, n_neuron), target_false)
        for s_target, s_label in zip(targets, labels):
            s_target[s_label] = target_true
        true_loss = np.sum(np.square(n_spike_per_neuron - targets), axis=1) / 2

        self.assertTrue(np.allclose(loss.get(), true_loss))

    def test_compute_errors(self):
        batch_size = 100
        n_neuron = 10
        max_spike = 15
        n_spike_per_neuron = np.random.randint(0, max_spike, size=(batch_size, n_neuron))
        n_spike_per_neuron_gpu = cp.array(n_spike_per_neuron, dtype=cp.float32)
        target_true = 10
        target_false = 2

        labels = np.random.randint(0, n_neuron, size=(batch_size,))
        labels_gpu = cp.array(labels, dtype=cp.int32)

        loss_fct = SpikeCountClassLoss(target_true=target_true, target_false=target_false)
        errors = loss_fct.compute_errors(cp.empty((batch_size, n_neuron, max_spike)),
                                         n_spike_per_neuron_gpu, labels_gpu)

        targets = np.full((batch_size, n_neuron), target_false)
        for s_target, s_label in zip(targets, labels):
            s_target[s_label] = target_true
        true_errors = np.repeat((targets - n_spike_per_neuron)[:, :, np.newaxis], max_spike, axis=2)

        self.assertTrue(np.allclose(errors.get(), true_errors))

    def test_compute_loss_and_errors(self):
        batch_size = 100
        n_neuron = 10
        max_spike = 15
        n_spike_per_neuron = np.random.randint(0, max_spike, size=(batch_size, n_neuron))
        n_spike_per_neuron_gpu = cp.array(n_spike_per_neuron, dtype=cp.float32)
        target_true = 10
        target_false = 2

        labels = np.random.randint(0, n_neuron, size=(batch_size,))
        labels_gpu = cp.array(labels, dtype=cp.int32)

        loss_fct = SpikeCountClassLoss(target_true=target_true, target_false=target_false)
        loss, errors = loss_fct.compute_loss_and_errors(cp.empty((batch_size, n_neuron, max_spike)),
                                                        n_spike_per_neuron_gpu, labels_gpu)

        targets = np.full((batch_size, n_neuron), target_false)
        for s_target, s_label in zip(targets, labels):
            s_target[s_label] = target_true
        true_loss = np.sum(np.square(n_spike_per_neuron - targets), axis=1) / 2
        true_errors = np.repeat((targets - n_spike_per_neuron)[:, :, np.newaxis], max_spike, axis=2)

        self.assertTrue(np.allclose(loss.get(), true_loss))
        self.assertTrue(np.allclose(errors.get(), true_errors))
