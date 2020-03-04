"""
extreme learning machine in python:
x1 x2 x3 x4
-   -   -   -   h = G(w*x + b)
h1  h2  h3
-   -   -       o = h^-1 * y  ???
o
"""
import numpy as np

class XLM:
    def __init__(self, no_hidden_units):
        self.no_hidden_units = no_hidden_units

    def fit(self, X, labels):
        """
        X is a 1-d array of inputs,
        labels are a 1-d array of outputs
        """
        # add a column of all ones to represent the biases
        X = np.column_stack([X, np.ones([X.shape[0]])])
        # weights and baises are all random numbers
        self.random_weights = np.random.randn(X.shape[1], self.no_hidden_units)
        # G = activation_function(weights * inputs + biases)
        G = self.activation_function(X.dot(self.random_weights))
        # the final layer is not random 
        self.w = np.linalg.pinv(G).dot(labels)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = self.activation_function(X.dot(self.random_weights))
        return G.dot(self.w)

    def __str__(self):
        return str(self.w)

    def activation_function(self, x):
        return self.softmax(x)

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def linear(self, x):
        return x

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)