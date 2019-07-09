import math

class LRScheduler:
    def __init__(self):
        self.constant(0.1)
        self.ty = 0
        self.iter = 1

    def reset():
        self.iter = 1

        
    def constant(self, lr0):
        self.lr0 = lr0
        self.ty = 0

    def time(self, lr0, decay):
        self.decay = decay
        self.lr0 = lr0
        self.ty = 1

    def step(self, lr0, step, decay):
        self.step = step
        self.decay = decay
        self.lr0 = lr0
        self.ty = 2
        
    
    def nextLR(self):
        if (self.ty == 0):
            lr = self.lr0
            self.iter += 1
            return lr
        elif (self.ty == 1):
            lr = self.lr0*(1 / (1. + self.decay * self.iter))
            self.iter += 1
            return lr
        elif (self.ty == 2):
            lr = self.lr0 * math.pow(self.decay, math.floor((self.iter-1)/self.step))
            self.iter += 1
            return lr
        
        return self.lr0
