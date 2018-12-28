import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



# Initialize model
model = TheModelClass()
model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Model's state_dict()")
print("This is a dictionary")
print(type(model.state_dict()))

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict()")
print("This is dictionary")
print(type(optimizer))

for optim_tensor in optimizer.state_dict():
    print(optim_tensor, "\t", optimizer.state_dict()[optim_tensor])

PATH = 'model.pt'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
model.train()
model.eval()
