from abc import abstractmethod, ABC
from typing import List
import cupy as cp


class AbstractOptimizer(ABC):
    def __init__(self, learning_rate: float):
        self._learning_rate: cp.float32 = cp.float32(learning_rate)

    @property
    def learning_rate(self) -> cp.float32:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate = cp.float32(value)

    @abstractmethod
    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        pass