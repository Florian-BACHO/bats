import cupy as cp

__compute_bias_gradient_kernel = cp.ElementwiseKernel('float32 time, float32 f2, float32 f3, float32 error, '
                                                      'float32 bias_scale',
                                                      'float32 out',
                                                      # Ternary condition to avoid nan values:
                                                      'out = (isinf(time)) ? (0.0) : (error * bias_scale * (f2 - f3))',
                                                      'compute_bias_gradient_kernel')


def compute_bias_gradient(spike_times: cp.array, f2: cp.ndarray, f3: cp.ndarray, errors: cp.ndarray,
                          bias_scale: cp.float32) -> cp.ndarray:
    bias_gradient = __compute_bias_gradient_kernel(spike_times, f2, f3, errors, bias_scale)
    bias_gradient = cp.sum(bias_gradient, axis=2)
    return bias_gradient
