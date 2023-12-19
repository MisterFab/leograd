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

class Tensor:
    def __init__(self, data, creator=None, ctx=None):
        self.data = data
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