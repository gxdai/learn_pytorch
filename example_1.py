import torch
import torch.nn as nn
from torch.autograd import Variable

# Defining Model
class LinearRegressor(nn.Module):
    def __init__(self, inp_size, require_bias=True):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(in_features=inp_size, out_features=1, bias=require_bias)
    
    def forward(self, inp_batch):
        return self.linear(inp_batch)
    
# Creating sample data for sum of 3 elements
X = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            X.append([i, j, k])

y = list(map(lambda x: x[0] + x[1] + x[2], X))
# X = Variable(torch.FloatTensor(X), requires_grad=False)
X = torch.FloatTensor(X)
# y = Variable(torch.FloatTensor(y), requires_grad=False)
y = torch.FloatTensor(y)
y = y.view(-1, 1)
# Creating model and defining loss function and optimizer
model = LinearRegressor(3)

# This is a non-parametric function
loss_fn = nn.MSELoss()
# So, we could use nn.functional.mse_loss(out, y) instead.
optim = torch.optim.Adam(model.parameters())

# Training the model
for epoch in range(10000):
    
    optim.zero_grad()
    
    out = model(X)
    # loss = loss_fn(out, y)
    loss = nn.functional.mse_loss(out, y)
    
    loss.backward()
    print("i = {:6d}, loss = {:5.3}".format(epoch, loss.item()))
    optim.step()
