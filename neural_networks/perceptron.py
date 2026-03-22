import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class Perceptron_2:
    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.zeros(X_bias.shape[1])

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X_bias, y):
                prediction = self.predict_single(xi)
                update = self.learning_rate * (target - prediction)
                if update != 0:
                    self.weights += update * xi
                    errors += 1
                if errors == 0:
                    break

    def net_input(self, X: np.ndarray) -> float:
        if self.weights is None:
            raise ValueError("Yolo")

        return np.dot(X, self.weights)

    def predict_single(self, xi: np.ndarray) -> int:
        return 1 if self.net_input(xi) >= 0.0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.where(self.net_input(X_bias) >= 0.0, 1, 0)


def plot_decision_boundary(model: Perceptron, X: np.ndarray, y: np.ndarray) -> None:
    # przygotuj zakres punktów
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    # rysowanie
    plt.contourf(
        xx, yy, Z, alpha=0.2, levels=[-0.1, 0.5, 1.1], colors=["#FFA07A", "#90EE90"]
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="bwr")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Granica decyzyjna perceptronu")
    plt.show()


if __name__ == "__main__":
    # X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y_train = np.array([0, 0, 0, 1])  # funkcja AND

    # model = Perceptron(learning_rate=0.1, n_epochs=100)
    # model.fit(X_train, y_train)

    # print("Wagi końcowe:", model.weights)
    # print("Predykcje:", model.predict(X_train))
    # plot_decision_boundary(model, X_train, y_train)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    clf = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    print("Wagi:\n", clf.coef_, "\nBias:", clf.intercept_)
    print("Dokładność:", accuracy_score(y, pred))