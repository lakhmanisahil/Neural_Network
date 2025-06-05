import numpy as np

class NeuralNet:
    def __init__(self, input_size=3072, hidden_size=256, output_size=3, lr=0.1):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size,))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size,))
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = X.dot(self.W1.T) + self.b1
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.W2.T) + self.b2
        probs = self.softmax(Z2)
        return Z1, A1, Z2, probs

    def compute_loss(self, probs, Y):
        N = probs.shape[0]
        correct_logprobs = -np.log(probs[np.arange(N), Y])
        return np.mean(correct_logprobs)

    def backward(self, X, Y, Z1, A1, probs):
        N = X.shape[0]
        dZ2 = probs
        dZ2[np.arange(N), Y] -= 1
        dZ2 /= N

        dW2 = dZ2.T.dot(A1)
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2.dot(self.W2)
        dZ1 = dA1 * (Z1 > 0)

        dW1 = dZ1.T.dot(X)
        db1 = np.sum(dZ1, axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=50):
        loss_history = []
        for epoch in range(epochs):
            Z1, A1, Z2, probs = self.forward(X)
            loss = self.compute_loss(probs, Y)
            loss_history.append(loss)
            self.backward(X, Y, Z1, A1, probs)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history

    def predict(self, X):
        _, _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)
