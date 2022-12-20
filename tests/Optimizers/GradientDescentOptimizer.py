import unittest
import numpy as np
import cupy as cp
from bats.Optimizers import GradientDescentOptimizer


class GradientDescentOptimizer_tests(unittest.TestCase):
    def test_gd(self):
        n_1 = 10
        n_2 = 5
        n_3 = 2
        n_epoch = 1000
        learning_rate = 0.05

        weights = [None,
                   cp.random.normal(loc=0.0, scale=1.0, size=(n_2, n_1), dtype=cp.float32),
                   cp.random.normal(loc=0.0, scale=1.0, size=(n_3, n_2), dtype=cp.float32)]

        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)

        for epoch in range(n_epoch):
            deltas = optimizer.step(weights)  # Optimize to 0
            for layer_weights, layer_deltas in zip(weights, deltas):
                if layer_weights is None:
                    continue
                for w, d in zip(layer_weights, layer_deltas):
                    w += d

        for layer_weights in weights:
            if layer_weights is None:
                continue
            for w in layer_weights:
                self.assertTrue(np.allclose(w.get(), np.zeros(w.shape)))
