import numpy as np
from .base import Optimizer

class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.v:
                self.v[0] = np.zeros_like(params[k])
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * (grads[k] ** 2)
            params[k] -= self.lr * grads[k] / (np.sqrt(self.v[k]) + self.eps)