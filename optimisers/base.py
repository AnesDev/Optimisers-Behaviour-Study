class Optimizer:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError