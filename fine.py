from __future__ import print_function, division
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

print("PyTorch version: ", torch.__version__)
print("Torchvision version: ", torchvision.__version__)


root_dir = './hymenoptera_data'
model_name = 'squeezenet'
num_classes = 2
batch_size = 8
num_epochs = 15
feature_extract = False
use_pretrained = False

device = torch.device('cuda:0')
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    
    for epoch in range(num_epochs):
        print("Epoch: [{:5d}/{:5d}]".format(epoch, num_epochs))
        print("-"*20)
        # each epoch has a train and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            # statics
            running_loss = 0.
            running_corrects = 0

            # Iterator over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero grads
                optimizer.zero_grad()
               
                # tracking grads if it is in training phase 
                with torch.set_grad_enabled(phase=='train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2 
                    else: 
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)                                
                running_corrects += torch.sum(preds == labels)    
                # running_corrects += torch.sum(preds == labels).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.item() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
           
            if phase == 'val' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)

    return model, val_acc_history

# set grad flag

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# tmp model
class NewResNet(nn.Module):
    def __init__(self):
        super(NewResNet, self).__init__()
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)
        self.resnet = models.resnet18()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1000)
        x = self.fc2(F.relu(self.fc1(x)))
        
        return x

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        """resnet18."""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_features, num_classes)
        input_size = 224

    elif model_name == 'alexnet':
        """AlexNet."""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(in_features, num_classes)
        input_size = 224
    elif model_name == 'vgg':
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(in_features, num_classes)
        input_size = 224
    elif model_name == 'squeezenet':
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == 'densenet':
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_features, num_classes)
        input_size = 224
    elif model_name == 'inception':
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        in_features = model_ft.fc.in_features
        aux_in_features = model_ft.AuxLogits.fc.in_features
        model_ft.fc = nn.Linear(in_features, num_classes)
        model_ft.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)
        input_size = 299
    else:
        print("Invalid model name, exiting ....")
        exit()

    return model_ft, input_size

num_classes = 2
"""
for model_name in ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']:
    print(model_name)
    model, size = initialize_model(model_name, num_classes, feature_extract)
    model.to(device)
    x = torch.randn(5, 3, size, size).to(device)
    y = model(x)
    print(y)
    print(model)
    print("Finish constructing model")
"""

# Load data
num_classes = 2
model_name = 'squeezenet'

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
model_ft.to(device)

print("Initializing Datasets and Dataloaders...")

data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(input_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.Resize(input_size),
                                              transforms.CenterCrop(input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
image_loader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x], batch_size=batch_size, 
                              shuffle=(True if x == 'train' else False),
                              num_workers=4) for x in ['train', 'val']} 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create optimizer
params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(name)  
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print(name)

optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# train and val
criterion = nn.CrossEntropyLoss()
model_ft, hist = train_model(model_ft, image_loader, criterion, 
                             optimizer, num_epochs=1000, 
                             is_inception=(model_name =='inception'))

# from scratch


