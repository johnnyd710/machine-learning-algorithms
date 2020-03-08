#!/usr/bin/env python

import numpy as np
import sys

def shuffle_split_data(X):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)

    train = X[split]
    test =  X[~split]

    return train, test

def one_hot_encode(data, mapping):
    X = data[:,1:]
    y = data[:, 0]

    y_one_hot = category_to_one_hot(y, mapping)
    return X, y_one_hot

def category_to_one_hot(a, mapping):
    """Convert an iterable of indices to one-hot encoded labels."""
    nb_classes = len(mapping.keys())
    map_y_to_int = list(map(lambda x: mapping[x], a))
    return np.eye(nb_classes)[list(map_y_to_int)]

def process(data):
    mapping = {key:i for key, i in enumerate(np.unique(data[:,0]))}
    train, test = shuffle_split_data(data)
    train_x, train_y = one_hot_encode(train, mapping)
    test_x, test_y = one_hot_encode(test, mapping)
    # train_x = np.divide(train_x, 255)
    # test_x = np.divide(test_x, 255)
    return train_x, train_y, test_x, test_y


# print(train_x)

# print(train_x.tobytes())