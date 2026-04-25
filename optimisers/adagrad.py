import numpy as np
from .base import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.g = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.G:
                self.g[k] = np.zeros_like(params[k])
            self.g[k] += grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.g[k]) + self.eps)