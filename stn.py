import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*4*4, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
            dtype=torch.float)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*3*3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
       
        return x

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def train(model, dataloaders, criterion, device, optimizer, num_epochs):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):

        for idx, (inputs, labels) in enumerate(dataloaders):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if idx % 500 == 0 and idx > 0:
                print("Epoch  [{:3d}/{:3d}] [{:3d}/{:3d}], Loss: {:5.3f}".\
                      format(idx, len(dataloaders), epoch, num_epochs,\
                             loss.item()))
    torch.save(model.state_dict(), 'mymodel.pt')

def test(model, dataloaders, device):
    model.load_state_dict(torch.load('mymodel.pt'))
    model.eval()
    acc = .0
    corrects = 0
    test_loss = .0
    with torch.set_grad_enabled(False):
        for idx, (inputs, labels) in enumerate(dataloaders):
            outputs = model(inputs)
            test_loss += F.nll_loss(outputs, labels, size_average=False).item()
            _, preds = torch.max(outputs, dim=1, keepdim=True) # _: value, preds: index
            
            corrects += torch.sum(preds==labels.view_as(preds)).item()
        print("\nTest set: Average loss: {:5.3f}, Accuracy: {}/{} ({:.0f}%))\n".format(test_loss / len(dataloaders.dataset), corrects, len(dataloaders.dataset), 100. * corrects/len(dataloaders.dataset)))

def convert_img_to_np(img):
    img = img.numpy().transpose((1, 2, 0))
    std = np.array([.229, 0.224, 0.225])
    mean = np.array([.485, 0.456, 0.406])
    img = img * std + mean
    
    return np.clip(img, 0, 1)

def visualize(model, test_loader, device):
    model = model.to(device)
    model.load_state_dict(torch.load('mymodel.pt'))
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()
        out_grid = convert_img_to_np(torchvision.utils.make_grid(transformed_input_tensor))
        
        in_grid = convert_img_to_np(torchvision.utils.make_grid(input_tensor))
        
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[1].imshow(out_grid)
        

        axarr[0].set_title("Org Image")
        axarr[1].set_title("Trans Image")

        

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.1307], [0.3081])])
    ), batch_size=64, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.1307], [0.3081])]
    )), batch_size=64, shuffle=True, num_workers=4)
    
    model = Net()
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # train(model, train_loader, criterion, device, optimizer, num_epochs)
    test(model, test_loader, device)
    visualize(model, test_loader, device)
    plt.ioff()
    plt.show()
