import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    """
    Computes the output of a linear model. (with one variable)
    :param x: an array of shape (n, 1) containing the input data
    :param w: a scalar value of the weight
    :param b: a scalar value of the bias
    :return:
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb



