from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=                  data_transforms[x]) for x in ['train', 'val']} 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
               shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# visualize
def imshow(inp, title=None):
    """Imshow for tensor."""
    inp = inp.numpy().transpose((1, 2, 0))   # convert it to numpy
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp * std + mean  # convert image back to original version
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)   # pause a bit so that plots are updated.


# get a batch of training data
# create a iterator and use next to get another batch

inputs, labels = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in labels])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
   

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-"*10)
        
        # Each epoch has a training and val phrase
        for phase in ['train', 'val']:
            if phase == 'train':
                # set model to training mode
                scheduler.step()
                model.train()
            else:
                # set model to eval mode
                model.eval()
            running_loss = .0
            running_corrects = 0 
            
            # iteration
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # track grad histroy only in training
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) 
                
                    # print(epoch, loss.item())
                    if phase  == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data) 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print("Training complete in {}m {}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val acc: {}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    model.save_state_dict('my.pt')
    return model

# visualize model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.load_state_dict('my.pt')
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
           
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        mode.train(mode=was_training)



# finetune
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
print(num_ftrs)
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay lr every 7 epochs by 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
# visualize_model(model_ft)



# only finetune the last layer


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.fc = torch.nn.Linear(model_conv.fc.in_features, 2)

