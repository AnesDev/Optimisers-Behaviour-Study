import numpy as np
from typing import Literal

class MLP:
    def __init__(self, input_size, hidden_size, output_size, optimiser: Literal["gradient_descent", "momentum", "adam", "RMSProp", "adagrad"] = "gradient_descent"):
        self.optimiser = optimiser
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.params = {
            "w1": np.random.randn(input_size, hidden_size),
            "b1": np.random.randn(1, hidden_size),
            "w2": np.random.randn(hidden_size, output_size),
            "b2": np.random.randn(1, output_size),
        }

        # Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        self.m_w1 = np.zeros_like(self.w1)
        self.v_w1 = np.zeros_like(self.w1)

        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)

        self.m_w2 = np.zeros_like(self.w2)
        self.v_w2 = np.zeros_like(self.w2)

        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)

        # Momentum
        self.beta = 0.9

        self.v_w1_m = np.zeros_like(self.w1)
        self.v_b1_m = np.zeros_like(self.b1)

        self.v_w2_m = np.zeros_like(self.w2)
        self.v_b2_m = np.zeros_like(self.b2)

        # AdaGrad
        self.G_w1 = np.zeros_like(self.w1)
        self.G_b1 = np.zeros_like(self.b1)
        self.G_w2 = np.zeros_like(self.w2)
        self.G_b2 = np.zeros_like(self.b2)

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy(self, A2, Y):
        epsilon = 1e-12
        m = Y.shape[0]
        return -np.sum(Y * np.log(A2 + epsilon)) / m

    def forward(self, X):
        Z1 = X @ self.w1 + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.w2 + self.b2
        A2 = self._softmax(Z2)
        return Z1, A1, Z2, A2
        
    def _backward_GD(self, X, Y, Z1, A1, A2, lr=0.001):
        dz2 = (A2 - Y) / X.shape[0]
        dW2 = A1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = dz2 @ self.w2.T * self.relu_derivative(Z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= lr * dW2
        self.b2 -= lr * db2
        self.w1 -= lr * dW1
        self.b1 -= lr * db1

    def _backward_momentum(self, X, Y, Z1, A1, A2, lr=0.001):
        m = X.shape[0]

        dz2 = (A2 - Y) / m
        dW2 = A1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = dz2 @ self.w2.T * self.relu_derivative(Z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.v_w2_m = self.beta * self.v_w2_m + (1 - self.beta) * dW2
        self.v_b2_m = self.beta * self.v_b2_m + (1 - self.beta) * db2
        self.v_w1_m = self.beta * self.v_w1_m + (1 - self.beta) * dW1
        self.v_b1_m = self.beta * self.v_b1_m + (1 - self.beta) * db1

        self.w2 -= lr * self.v_w2_m
        self.b2 -= lr * self.v_b2_m
        self.w1 -= lr * self.v_w1_m
        self.b1 -= lr * self.v_b1_m

    def _backward_adam(self, X, Y, Z1, A1, A2, lr=0.001):
        self.t += 1
        m = X.shape[0]

        dz2 = A2 - Y
        dw2 = (A1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * self.relu_derivative(Z1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * dw2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * dw1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1

        self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (dw2 ** 2)
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (dw1 ** 2)
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)

        m_w2_hat = self.m_w2 / (1 - self.beta1 ** self.t)
        m_b2_hat = self.m_b2 / (1 - self.beta1 ** self.t)
        v_w2_hat = self.v_w2 / (1 - self.beta2 ** self.t)
        v_b2_hat = self.v_b2 / (1 - self.beta2 ** self.t)

        m_w1_hat = self.m_w1 / (1 - self.beta1 ** self.t)
        m_b1_hat = self.m_b1 / (1 - self.beta1 ** self.t)
        v_w1_hat = self.v_w1 / (1 - self.beta2 ** self.t)
        v_b1_hat = self.v_b1 / (1 - self.beta2 ** self.t)

        self.w2 -= lr * m_w2_hat / (np.sqrt(v_w2_hat) + self.eps)
        self.b2 -= lr * m_b2_hat / (np.sqrt(v_b2_hat) + self.eps)
        self.w1 -= lr * m_w1_hat / (np.sqrt(v_w1_hat) + self.eps)
        self.b1 -= lr * m_b1_hat / (np.sqrt(v_b1_hat) + self.eps)

    def _backward_RMSProp(self, X, Y, Z1, A1, A2, lr=0.001):
        m = X.shape[0]

        dz2 = A2 - Y
        dw2 = (A1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * self.relu_derivative(Z1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (dw2 ** 2)
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (dw1 ** 2)
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)

        self.w2 -= lr * dw2 / (np.sqrt(self.v_w2) + self.eps)
        self.b2 -= lr * db2 / (np.sqrt(self.v_b2) + self.eps)
        self.w1 -= lr * dw1 / (np.sqrt(self.v_w1) + self.eps)
        self.b1 -= lr * db1 / (np.sqrt(self.v_b1) + self.eps)

    def _backward_adagrad(self, X, Y, Z1, A1, A2, lr=0.01):
        m = X.shape[0]

        dz2 = A2 - Y
        dw2 = (A1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = dz2 @ self.w2.T * self.relu_derivative(Z1)
        dw1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.G_w2 += dw2 ** 2
        self.G_b2 += db2 ** 2
        self.G_w1 += dw1 ** 2
        self.G_b1 += db1 ** 2

        self.w2 -= lr * dw2 / (np.sqrt(self.G_w2) + self.eps)
        self.b2 -= lr * db2 / (np.sqrt(self.G_b2) + self.eps)
        self.w1 -= lr * dw1 / (np.sqrt(self.G_w1) + self.eps)
        self.b1 -= lr * db1 / (np.sqrt(self.G_b1) + self.eps)

    def backward(self, X, Y, Z1, A1, A2, lr=0.001):
        if self.optimiser == "gradient_descent":
            self._backward_GD(X, Y, Z1, A1, A2, lr)
        elif self.optimiser == "momentum":
            self._backward_momentum(X, Y, Z1, A1, A2, lr)
        elif self.optimiser == "adam":
            self._backward_adam(X, Y, Z1, A1, A2, lr)
        elif self.optimiser == "RMSProp":
            self._backward_RMSProp(X, Y, Z1, A1, A2, lr)
        elif self.optimiser == "adagrad":
            self._backward_adagrad(X, Y, Z1, A1, A2, lr)
        else:
            raise ValueError(f"Unknown optimiser: {self.optimiser}")

    def train(self, X, Y, epochs=1000, lr=0.01):
        self.t = 0
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward(X)
            loss = self._cross_entropy(A2, Y)
            self.backward(X, Y, Z1, A1, A2, lr)
            if i % 100 == 0:
                print(f"Epoch {i}, Loss:{loss:.4f}")

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=1)