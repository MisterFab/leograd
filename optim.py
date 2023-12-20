class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
    
    def step(self):
        for param in self.parameters:
            param.data -= param.grad * self.lr