import numpy as np
from utils.sigmoid import sigmoid

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init the weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(y_linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
        

    def predict(self, X):
        y_linear = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(y_linear)
        y_pred_classes = [1 if x > 0.5 else 0 for x in y_pred]
        return y_pred_classes