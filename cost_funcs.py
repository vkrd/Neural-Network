import numpy as np

def MSE(x, y):
    return np.mean(0.5*np.square(y-x))

def d_MSE(x, y):
    return (x-y)
