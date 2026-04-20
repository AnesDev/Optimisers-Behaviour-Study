from base import Optimizer

class GradientDescent(Optimizer):
    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]