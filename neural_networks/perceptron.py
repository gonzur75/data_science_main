import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Dodajemy kolumnę biasu
        self.weights = np.zeros_like(X_bias.shape[1])

        for _ in range(self.n_epochs):
            errors = 0

            for xi, target in zip(X, y):
                pass


if __name__ == "__main__":
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])  # funkcja AND

    model = Perceptron(learning_rate=0.1, n_epochs=10)
    model.fit(X_train, y_train)