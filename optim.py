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

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0] * len(parameters)
        self.v = [0] * len(parameters)
        self.t = 0

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            assert param.grad is not None
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)