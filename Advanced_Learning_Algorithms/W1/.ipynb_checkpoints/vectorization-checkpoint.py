"""
    One of the reasons that deep learning researchers have been able to scale up neural networks, 
    and thought really large neural networks over the last decade, is because neural networks can be vectorized. 
    
    They can be implemented very efficiently using matrix multiplications. 
    
    It turns out that parallel computing hardware, including GPUs, but also some CPU functions are very good at doing very large matrix multiplications. 
    
    In file, we'll take a look at how these vectorized implementations of neural networks work.
"""

# forward propagation in a single layer of a neural network (Using a Loop)

import numpy as np
from forward_prop_numpy import g

x = np.array([200, 17])
W = np.array([[1, -3, 5],
             [-2, 4, -6]])
b = np.array([-1, 1, 2])

def dense(a_in, W, b):
    """
        Computes dense layer
        Args:
          a_in (ndarray (n, )) : Data, 1 example 
          W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
          b    (ndarray (j, )) : bias vector, j units  
        Returns
          a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out                                     # eg. [1, 0, 1]


# forward propagation in a single layer of a neural network (vectorization/ using matrices)

# weights and input are all 2D
x = np.array([[200, 17]])    # a 2D array
W = np.array([[1, -3, 5],
             [-2, 4, -6]])
b = np.array([[-1, 1, 2]])   # a 2D array

def dense(A_in, W, B):
    Z = np.matmul(A_in, W) + B       # matmul = matrix multiplication
    A_out = g(Z)
    return A_out                                   # eg. [[1, 0, 1]]
