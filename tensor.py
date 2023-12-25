import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors = ()
    
    def __repr__(self):
        return f"Context({self.saved_tensors})"
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def get_saved_tensors(self):
        return self.saved_tensors

class Function:
    def __call__(self, *args, **kwargs):
        context = Context()
        result = self.forward(context, *args, **kwargs)
        return Tensor(result, self, context)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Tensor:
    def __init__(self, data, creator=None, context=None, dtype=np.float32):
        if not isinstance(data, (np.ndarray, list, int, float)):
            raise TypeError("Data must be a numpy.ndarray or a list.")
        self.data = np.array(data, dtype=dtype) if isinstance(data, list) else data
        self.grad = None
        self.creator = creator
        self.context = context
    
    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            multiplication_function = Multiply()
            return multiplication_function(self, other)
        else:
            return Tensor(self.data * other)

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = 1 if self.grad is None else self.grad

        if self.creator:
            gradients = self.creator.backward(self.context, grad_output)
            if not isinstance(gradients, tuple):
                gradients = (gradients,)

            for gradient, input_tensor in zip(gradients, self.context.get_saved_tensors()):
                input_tensor.backward(gradient)
        else:
            self.grad = grad_output if self.grad is None else self.grad + grad_output
    
    def relu(self):
        f = ReLU()
        return f(self)
    
    def sigmoid(self):
        f = Sigmoid()
        return f(self)
    
    def reshape(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return Tensor(np.reshape(self.data, new_shape))
    
    @staticmethod
    def kaiming_uniform(input_dim, output_dim):
        limit = np.sqrt(6 / input_dim)
        return np.random.uniform(low=-limit, high=limit, size=(input_dim, output_dim))

class Multiply(Function):
    @staticmethod
    def forward(ctx, tensor_a, tensor_b):
        ctx.save_for_backward(tensor_a, tensor_b)
        return tensor_a.data * tensor_b.data

    @staticmethod
    def backward(ctx, grad_output):
        tensor_a, tensor_b = ctx.get_saved_tensors()
        grad_tensor_a = grad_output * tensor_b.data
        grad_tensor_b = grad_output * tensor_a.data
        return grad_tensor_a, grad_tensor_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return np.maximum(0, input_tensor.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.get_saved_tensors()
        return grad_output * (input_tensor.data > 0)
    
class Sigmoid(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        sigmoid_input = np.clip(input_tensor.data, -100, 100)
        return 1 / (1 + np.exp(-sigmoid_input))
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.get_saved_tensors()
        sigmoid_output = 1 / (1 + np.exp(-np.clip(input_tensor.data, -100, 100)))
        return grad_output * sigmoid_output * (1 - sigmoid_output)

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight_tensor, bias_tensor=None):
        ctx.save_for_backward(input_tensor, weight_tensor, bias_tensor)
        output = input_tensor.data.dot(weight_tensor.data)
        if bias_tensor is not None:
            output += bias_tensor.data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight_tensor, bias_tensor = ctx.get_saved_tensors()
        grad_input = grad_output.dot(weight_tensor.data.T)
        grad_weight = input_tensor.data.T.dot(grad_output)
        grad_bias = grad_output.sum(axis=0) if bias_tensor is not None else None
        return grad_input, grad_weight, grad_bias

class MeanSquaredError(Function):
    @staticmethod
    def forward(ctx, input_tensor, target_tensor):
        ctx.save_for_backward(input_tensor, target_tensor)
        error_margin = 1e-8
        return np.square(input_tensor.data - target_tensor.data + error_margin).mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, target_tensor = ctx.get_saved_tensors()
        grad = 2 * (input_tensor.data - target_tensor.data) / input_tensor.data.size
        grad_input = grad * grad_output
        grad_target = -grad_input
        return grad_input, grad_target

class BinaryCrossEntropy(Function):
    @staticmethod
    def forward(ctx, input_tensor, target_tensor):
        epsilon = 1e-12
        input_tensor.data = np.clip(input_tensor.data, epsilon, 1 - epsilon)
        ctx.save_for_backward(input_tensor, target_tensor)
        return -(target_tensor.data * np.log(input_tensor.data) + (1 - target_tensor.data) * np.log(1 - input_tensor.data)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, target_tensor = ctx.get_saved_tensors()
        grad_input = (input_tensor.data - target_tensor.data) / (input_tensor.data * (1 - input_tensor.data)) * grad_output
        grad_target = -grad_input
        return grad_input, grad_target
 
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        input_max = np.max(input_tensor.data, axis=1, keepdims=True)
        shifted_inputs = input_tensor.data - input_max
        exp_shifted = np.exp(shifted_inputs)
        log_sum_exp = np.log(np.sum(exp_shifted, axis=1, keepdims=True))
        return shifted_inputs - log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.get_saved_tensors()
        softmax_output = np.exp(input_tensor.data - np.max(input_tensor.data, axis=1, keepdims=True))
        softmax_output /= np.sum(softmax_output, axis=1, keepdims=True)
        return grad_output - softmax_output * np.sum(grad_output, axis=1, keepdims=True)

class NLLLoss(Function):
    @staticmethod
    def forward(ctx, input_tensor, target):
        ctx.save_for_backward(input_tensor, target)
        correct_log_probs = input_tensor.data[np.arange(target.data.shape[0]), target.data]
        return -np.mean(correct_log_probs)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, target = ctx.get_saved_tensors()
        grad_input = np.zeros_like(input_tensor.data)
        grad_input[np.arange(target.data.shape[0]), target.data] = -1 / target.data.shape[0] * grad_output
        return grad_input, target   

class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input_tensor, target_tensor):
        f = LogSoftmax()
        log_softmax = f(input_tensor)
        f = NLLLoss()
        loss = f(log_softmax, target_tensor)
        ctx.save_for_backward(loss)
        return loss.data
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.get_saved_tensors()
        grad_input = input_tensor.data * grad_output
        return grad_input