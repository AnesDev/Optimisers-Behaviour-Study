import numpy as np
from .base import Optimizer

class Momentum(Optimizer):
    def __init__(self, lr=0.001, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.velocity:
                self.velocity[k] = np.zeros_like(params[k])
            self.velocity[k] = self.beta * self.velocity[k] + (1 - self.beta) * grads[k]
            params[k] -= self.lr * self.velocity[k]