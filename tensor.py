import numpy as np

class AutogradContext:
    def __init__(self):
        self.saved_tensors = ()
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def get_saved_tensors(self):
        return self.saved_tensors

class Function:
    def __call__(self, *args, **kwargs):
        ctx = AutogradContext()
        result = self.forward(ctx, *args, **kwargs)
        return Tensor(result, self, ctx)
    
class Multiply(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data * b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.get_saved_tensors()
        grad_a = grad_output * b.data
        grad_b = grad_output * a.data
        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp.data * (inp.data > 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.get_saved_tensors()
        return grad_output * (inp.data > 0)

class Tensor:
    def __init__(self, data, creator=None, ctx=None):
        self.data = np.array(data) if isinstance(data, list) else data
        self.grad = None
        self.creator = creator
        self.ctx = ctx
    
    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            f = Multiply()
            return f(self, other)
        else:
            return Tensor(self.data * other)

    def backward(self, grad_output=None):
        if grad_output is None:
            if self.grad is None:
                grad_output = 1
            else:
                grad_output = self.grad

        if self.creator:
            grads = self.creator.backward(self.ctx, grad_output)
            if not isinstance(grads, tuple):
                grads = (grads,)

            for grad, inp in zip(grads, self.ctx.get_saved_tensors()):
                inp.backward(grad)
        else:
            if self.grad is None:
                self.grad = grad_output
            else:
                self.grad += grad_output
    
    def relu(self):
        f = ReLU()
        return f(self)
    
    @staticmethod
    def kaiming_uniform(n_inputs, n_outputs):
        limit = np.sqrt(6 / n_inputs)
        return np.random.uniform(low=-limit, high=limit, size=(n_inputs, n_outputs))

class Multiply(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data * b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.get_saved_tensors()
        grad_a = grad_output * b.data
        grad_b = grad_output * a.data
        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp.data * (inp.data > 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.get_saved_tensors()
        return grad_output * (inp.data > 0)

class Linear(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias = None):
        ctx.save_for_backward(inp, weight, bias)
        output = inp.data.dot(weight.data)
        if bias is not None:
            output += bias.data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.get_saved_tensors()
        grad_inp = grad_output.dot(weight.data.T)
        grad_weight = inp.data.T.dot(grad_output)
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(axis=0)
        return grad_inp, grad_weight, grad_bias

class MeanSquaredError(Function):
    @staticmethod
    def forward(ctx, inp, target):
        ctx.save_for_backward(inp, target)
        return np.square(inp.data - target.data).mean()

    @staticmethod
    def backward(ctx, grad_output):
        inp, target = ctx.get_saved_tensors()
        grad_inp = 2 * (inp.data - target.data) / inp.data.size
        grad_target = -grad_inp
        return grad_inp, grad_target