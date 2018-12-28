import torch
import random

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct there are nn.Linear instances
        """
        super(DynamicNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device("cuda:3")
dtype = torch.float
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

model = DynamicNet(D_in, H, D_out)
model = model.to(device)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

