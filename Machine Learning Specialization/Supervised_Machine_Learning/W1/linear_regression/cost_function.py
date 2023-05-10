from model_repr import compute_model_output

def compute_cost(x, y, w, b,):
    """
    Computes the cost of a linear regression model with one variable.
    :param x: an array of shape (n, 1) containing the input data
    :param y: an array of shape (n, 1) containing the output data
    :param w: a scalar value of the weight
    :param b: a scalar value of the bias
    :return: the cost of the model
    """
    # number of training examples
    m = x.shape[0]


    J = 0

    # predicted outputs
    f_wb = compute_model_output(x, w, b)

    # sum of squared errors
    for i in range(m):
        J += (f_wb[i] - y[i]) ** 2

    # average of squared errors
    J /= (2 * m)

    return J