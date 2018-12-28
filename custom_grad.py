import torch

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd functions by subclassing 
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """
    # this works like for class, not object. 
    @staticmethod  
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a tensor containing output. ctx is context object that can be used to stash information for backward computation. 
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss with respect to the output, and we need to compute the gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input



dtype = torch.float
device = torch.device('cuda:3')

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(1000):
    # To apply my own function, we use function.apply method
    relu = MyReLU.apply
    y_pred = relu(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    loss.backward()
    # update
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
