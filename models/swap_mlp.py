import numpy as np

class swap_MLP:
    def __init__(self, input_size, hidden_size, output_size, optimizer, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.optimizer = optimizer
        self.params = {
            "w1": np.random.randn(input_size, hidden_size),
            "b1": np.random.randn(1, hidden_size),
            "w2": np.random.randn(hidden_size, output_size),
            "b2": np.random.randn(1, output_size),
        }

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy(self, A2, Y):
        epsilon = 1e-12
        m = Y.shape[0]
        return -np.sum(Y * np.log(A2 + epsilon)) / m

    def forward(self, X):
        self.Z1 = X @ self.params["w1"] + self.params["b1"]
        self.A1 = self.relu(self.Z1)

        i = np.random.randint(0, 20)
        j = np.random.randint(0, 20)
        
        while(i == j):
            i = np.random.randint(0, 20)
            j = np.random.randint(0, 20)
            
        
        self.A1[i], self.A1[j] = self.A1[j], self.A1[i]

        self.Z2 = self.A1 @ self.params["w2"] + self.params["b2"]
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self, X, Y):
        m = X.shape[0]
        dz2 = (self.A2 - Y) / m
        dW2 = self.A1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = dz2 @ self.params["w2"].T * self.relu_derivative(self.Z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return {
            "w1": dW1,
            "b1": db1, 
            "w2": dW2, 
            "b2": db2
        }
    
    def train(self, X, Y, epochs=1000):
        loss_history = []

        for i in range(epochs):
            self.forward(X)
            grads = self.backward(X, Y)
            self.optimizer.step(self.params, grads)
            if i % 100 == 0:
                loss = self.cross_entropy(self.A2, Y)
                loss_history.append(loss)

        return loss_history

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)