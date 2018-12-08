"""Data Utility Functions."""
# pylint: disable=invalid-name
import os
import pickle as pickle

import numpy as np


def load_cifar_batch(filename):
    """Load single batch of CIFAR-10."""
    with open(filename, 'rb') as f:
        # load with encoding because file was pickled with Python 2
        data_dict = pickle.load(f, encoding='latin1')
        X = np.array(data_dict['data'])
        Y = np.array(data_dict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X, Y


def load_CIFAR10(root_dir):
    """Load all of CIFAR-10."""
    f = os.path.join(root_dir, 'cifar10_train.p')
    X_batch, y_batch = load_cifar_batch(f)
    return X_batch, y_batch


def scoring_function(x, lin_exp_boundary, doubling_rate):
    """Computes score function values.

        The scoring functions starts linear and evolves into an exponential
        increase.
    """
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x < lin_exp_boundary)
    exp_region = np.logical_and(x >= lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score
