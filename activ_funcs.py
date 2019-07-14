import numpy as np

def sigmoid(x):
    # print((1/(1+np.exp(-x))).shape)
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    # print((x*(1-x)).shape)
    return x * (1 - x)

def tanh(x):
    a = np.exp(x)
    b = np.exp(-x)
    return np.divide(a - b, a + b)

def d_tanh(x):
    return 1 - np.square(tanh(x))

def ReLU(x):
    return np.maximum(x, np.zeros(x.shape))

def d_ReLU(x):
    return np.where(x <= 0, 0, 1)

def leaky_ReLU(x):
    return np.where(x > 0, x, x*0.01)

def d_leaky_ReLU(x):
    return np.where(x > 0, 1, 0.01)
