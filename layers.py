from activ_funcs import *

class Dense:
    def __init__(self, nodes, name="d", activation="sig"):
        self.nodes = nodes
        self.name = name
        self.activ_name = activation

    def activation(self, x):
        if (self.activ_name == "sig"):
            return sigmoid(x)
        elif (self.activ_name == "tanh"):
            return tanh(x)
        elif (self.activ_name == "relu"):
            return ReLU(x)
        
        else:
            #By default use sigmoid
            return sigmoid(x)
        
    def d_activation(self, x):
        if (self.activ_name == "sig"):
            return d_sigmoid(x)

        elif (self.activ_name == "tanh"):
            return d_tanh(x)
        elif (self.activ_name == "relu"):
            return d_ReLU(x)
        else:
            #By default use derivative of sigmoid
            return d_sigmoid(x)
