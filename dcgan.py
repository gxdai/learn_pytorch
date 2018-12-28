from __future__ import print_function

import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML






manual_seed = 2222
print("Random seed: {}".format(manual_seed))
random.seed(manual_seed)
torch.manual_seed(manual_seed)
"""
dataroot: the path to the root of the dataset folder.
workers: the number of worker threads for loading the data with DataLoader.
batch_size:
image_size:
nc:
nz:
ngf:

num_epochs:
lr
beta1:
ngpu:
"""


dataroot = './facades'
workers = 4
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = 2

dataset = datasets.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# visualize image
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.title('Training Image')
plt.axis('off')
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
 

# reinitialize all the conv layers
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), 
                                  nn.BatchNorm2d(ngf*8),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), 
                                  nn.BatchNorm2d(ngf*4),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), 
                                  nn.BatchNorm2d(ngf*2),
                                  nn.ReLU(True),  
                                  nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), 
                                  nn.BatchNorm2d(ngf),
                                  nn.ReLU(True),
                                  # final conv to image
                                  nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), 
                                  nn.Tanh(),
                                  )
    def forward(self, input):
        return self.main(input)



# reinitialize generator with weight_init
netG = Generator(ngpu=ngpu).to(device)

if (device.type == 'cuda') and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# reinitialize weights
netG.apply(weight_init)
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.main(input)


netD = Discriminator(ngpu).to(device)

if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weight_init)
print(netD)


criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0


optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
D_losses = []
G_losses = []

iters = 0


print("Training")
for epoch in range(num_epochs):
    print(len(dataloader))
    for i, data in enumerate(dataloader):
        # update D
        optimizerD.zero_grad()
        inputs = data[0].to(device)
        b_size = inputs.size(0)
        label = torch.full((b_size,), real_label, device=device)
        outputs = netD(inputs).view(-1)
        D_loss_real = criterion(outputs, label.view(-1))
        D_loss_real.backward()
        D_x = D_loss_real.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        outputs = netD(fake.detach()).view(-1)
        label.fill_(fake_label)
        D_loss_fake = criterion(outputs, label.view(-1))
       
        D_loss_fake.backward()
        D_x_f = outputs.mean().item()
        D_loss = D_loss_real + D_loss_fake
        optimizerD.step()

        # update G
        netG.zero_grad()
        label.fill_(real_label)
        outputs = netD(fake).view(-1)
        G_loss = criterion(outputs, label)
        G_loss.backward()
        G_x = outputs.mean().item()

        
        if i % 50 == 0:
            print('[%d/%d] [%d/%d]\t Loss_D: %.4f\tLoss_G: %.4f\tD(x):\
                   %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader),
                   D_loss.item(), D_loss.item(), D_x, D_x_f, G_x))

        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())
       
        if iters % 500 == 0 or (epoch == num_epochs-1 and i == len(dataloader) - 1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True)) 
            iters += 1


        
   
