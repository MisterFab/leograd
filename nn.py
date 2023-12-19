from tensor import Tensor, Linear

import numpy as np

class Module:
    def __init__(self):
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        params = []
        for name, module in self._modules.items():
            params.extend(module.parameters())
        return params
    
    def __call__(self, *args):
        return self.forward(*args)

class Layer:
    def __call__(self, *args):
        return self.forward(*args)

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.weights = Tensor(Tensor.kaiming_uniform(n_inputs, n_outputs))
        self.bias = Tensor(np.zeros(n_outputs))
    
    def forward(self, inp):
        f = Linear()
        return f(inp, self.weights, self.bias)
    
    def parameters(self):
        return [self.weights, self.bias]