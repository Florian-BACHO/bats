from typing import Optional, List
import cupy as cp

from ..AbstractOptimizer import AbstractOptimizer


class GradientDescentOptimizer(AbstractOptimizer):
    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        return [None if g is None else -self._learning_rate * g for g in gradient]
