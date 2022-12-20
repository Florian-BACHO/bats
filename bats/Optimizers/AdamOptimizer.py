from typing import Optional, List
import cupy as cp
import numpy as np

from ..AbstractOptimizer import AbstractOptimizer

update_m_kernel = cp.ElementwiseKernel("float32 m, float32 beta_1, float32 one_minus_beta_1, float32 grad",
                                       "float32 new_m",
                                       "new_m = beta_1 * m + one_minus_beta_1 * grad",
                                       "update_m_kernel")
update_v_kernel = cp.ElementwiseKernel("float32 v, float32 beta_2, float32 one_minus_beta_2, float32 grad",
                                       "float32 new_v",
                                       "new_v = beta_2 * v + one_minus_beta_2 * grad * grad",
                                       "update_v_kernel")
compute_deltas_kernel = cp.ElementwiseKernel("float32 grad, float32 m_hat, float32 v_hat, float32 learning_rate,"
                                             "float32 epsilon",
                                             "float32 delta",
                                             "delta = -(learning_rate * m_hat / (sqrtf(v_hat) + epsilon))",
                                             "compute_deltas_kernel")


class AdamOptimizer(AbstractOptimizer):
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.__beta_1: cp.float32 = cp.float32(beta_1)
        self.__one_minus_beta_1: cp.float32 = cp.float32(1.0 - beta_1)
        self.__beta_2: cp.float32 = cp.float32(beta_2)
        self.__one_minus_beta_2: cp.float32 = cp.float32(1.0 - beta_2)
        self.__epsilon: cp.float32 = cp.float32(epsilon)

        self.__m: Optional[List[List[cp.array]]] = None
        self.__v: Optional[List[List[cp.array]]] = None
        self.__t: cp.int32 = cp.int32(0)

    def step(self, gradient: List[cp.ndarray]) -> List[cp.ndarray]:
        self.__t += 1

        # Set m and v to 0 at first iteration
        if self.__m is None:
            self.__m = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient]
            self.__v = [None if grad is None else cp.zeros(grad.shape, dtype=cp.float32) for grad in gradient]
        # Update m and v
        self.__m = [None if grad is None else update_m_kernel(pre_m, self.__beta_1, self.__one_minus_beta_1, grad)
                    for pre_m, grad in zip(self.__m, gradient)]
        self.__v = [None if grad is None else update_v_kernel(pre_v, self.__beta_2, self.__one_minus_beta_2, grad)
                    for pre_v, grad in zip(self.__v, gradient)]

        # Compute m_hat and v_hat
        one_minus_beta_1_power_t = 1 - self.__beta_1 ** self.__t
        one_minus_beta_2_power_t = 1 - self.__beta_2 ** self.__t
        m_hat = [None if m is None else m / one_minus_beta_1_power_t for m in self.__m]
        v_hat = [None if v is None else v / one_minus_beta_2_power_t for v in self.__v]

        return [None if g is None else compute_deltas_kernel(g, m, v, self._learning_rate, self.__epsilon)
                for g, m, v in zip(gradient, m_hat, v_hat)]
