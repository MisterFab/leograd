import torch
import numpy as np
from tensor import Tensor
import time

a_vals = [2.0, -1.0, 0.0, 3.0, -2.0, 4.0, -3.5, 1.5, -4.5, 0.5]
b_vals = [3.0, -1.0, 0.0, -2.0, 2.0, -4.0, 2.5, -1.5, 3.5, -0.5]

start = time.time()
a_custom = Tensor(a_vals)
b_custom = Tensor(b_vals)
output_custom = a_custom * b_custom
output_relu_custom = output_custom.relu()
output_relu_custom.backward()
end = time.time()
print("Custom time:", end - start)

start = time.time()
a_torch = torch.tensor(a_vals, requires_grad=True)
b_torch = torch.tensor(b_vals, requires_grad=True)
output_torch = a_torch * b_torch
output_relu_torch = torch.relu(output_torch)
grad_output = torch.ones_like(output_relu_torch)
output_relu_torch.backward(grad_output)
end = time.time()
print("Torch time:", end - start)

# Comparing the outputs and gradients
outputs_equal = np.allclose(output_relu_custom.data, output_relu_torch.detach().numpy())
print("Outputs are equal:", outputs_equal)

gradients_equal_a = np.allclose(a_custom.grad, a_torch.grad.numpy())
gradients_equal_b = np.allclose(b_custom.grad, b_torch.grad.numpy())
print("Gradients for 'a' are equal:", gradients_equal_a)
print("Gradients for 'b' are equal:", gradients_equal_b)