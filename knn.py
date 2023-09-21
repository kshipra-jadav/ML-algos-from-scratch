import numpy as np
from collections import Counter
from utils.eucilidian_distance import eucilidian_distance

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # calculate the eucilidian distance
        distances = [eucilidian_distance(x, y) for y in self.X_train]        

        # get the k-nearest neighbors
        nearest_indicies = np.argsort(distances)[0 : self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indicies]

        # calculate the majority label
        majority_labels = Counter(nearest_labels).most_common(1)
        return majority_labels[0][0]

