import numpy as np

def MSE(x, y):
    return np.mean(np.square(y-x))

def d_MSE(x, y):
    return 2*(y-x)
