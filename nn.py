import torch

device = torch.device('cuda:3')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as sequence of layers.
# nn.sequential is a module which contains other modules, and applies them in sequence to produce its output.

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
model = model.to(device)
# Loss
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500000):
    # override the __call__ operator
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # zero gradients before running backward pass.
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()
