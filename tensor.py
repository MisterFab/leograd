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

class Tensor:
    def __init__(self, data, creator=None, ctx=None, dtype=np.float32):
        self.data = np.array(data, dtype=dtype) if isinstance(data, list) else data
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
    
    def sigmoid(self):
        f = Sigmoid()
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

class ReLU(Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp.data * (inp.data > 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.get_saved_tensors()
        return grad_output * (inp.data > 0)
    
class Sigmoid(Function):
    @staticmethod
    def forward(ctx, inp):
        inp_safe = np.clip(inp.data, -100, 100)
        out = 1 / (1 + np.exp(-inp_safe))
        ctx.save_for_backward(inp)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.get_saved_tensors()
        sigmoid_output = 1 / (1 + np.exp(-np.clip(inp.data, -100, 100)))
        return grad_output * sigmoid_output * (1 - sigmoid_output)

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
        epsilon = 1e-8
        return np.square(inp.data - target.data + epsilon).mean()

    @staticmethod
    def backward(ctx, grad_output):
        inp, target = ctx.get_saved_tensors()
        grad = 2 * (inp.data - target.data) / inp.data.size
        grad_inp = grad * grad_output
        grad_target = -grad_inp
        return grad_inp, grad_target

class BinaryCrossEntropy(Function):
    @staticmethod
    def forward(ctx, inp, target):
        ctx.save_for_backward(inp, target)
        epsilon = 1e-12
        inp_safe = np.clip(inp.data, epsilon, 1 - epsilon)
        return -(target.data * np.log(inp_safe) + (1 - target.data) * np.log(1 - inp_safe)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        inp, target = ctx.get_saved_tensors()
        epsilon = 1e-12
        inp_safe = np.clip(inp.data, epsilon, 1 - epsilon)
        grad_inp = (inp_safe - target.data) / (inp_safe * (1 - inp_safe)) / inp.data.size
        grad_target = -grad_inp
        return grad_inp, grad_target