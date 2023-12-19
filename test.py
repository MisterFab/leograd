import torch
from tensor import Tensor

a_val = 2.0
b_val = 3.0

a_custom = Tensor(a_val)
b_custom = Tensor(b_val)
output_custom = a_custom * b_custom
output_custom.backward()

a_torch = torch.tensor(a_val, requires_grad=True)
b_torch = torch.tensor(b_val, requires_grad=True)
output_torch = a_torch * b_torch
output_torch.backward()

outputs_equal = output_custom.data == output_torch.item()
print("Outputs are equal:", outputs_equal)

gradients_equal_a = a_custom.grad == a_torch.grad.item()
gradients_equal_b = b_custom.grad == b_torch.grad.item()
print("Gradients for 'a' are equal:", gradients_equal_a)
print("Gradients for 'b' are equal:", gradients_equal_b)