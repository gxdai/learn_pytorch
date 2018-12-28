import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        # non parametric function torch.sigmiod()
        return torch.sigmoid(self.linear(x))

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.linear = nn.Linear(10, 2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.linear(x))


torch.manual_seed(222)
model1 = Model1()
torch.manual_seed(222)
model2 = Model2()

print(model1)
print(model2)

x = torch.randn(3, 10)
print(model1(x))
print(model2(x))
