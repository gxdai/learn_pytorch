from __future__ import print_function
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

imsize = (512, 512) if torch.cuda.is_available() else (128, 128)

img_trans = transforms.Compose([transforms.Resize(imsize),
                                transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name) # this image could directly go to trans
    img_tensor = img_trans(image)
    img_tensor = img_tensor.unsqueeze(0)
    # move to gpu
    return img_tensor.to(device, torch.float)


style_img = image_loader('./images/picasso.jpg')
content_img = image_loader('./images/dancing.jpg')

assert style_img.size() == content_img.size(), \
    "we need both images have the same size"


# visualize the images for verification.    
unloader = transforms.ToPILImage() # convert it into PIL image

plt.ion()  # turn on interactive mode


def imshow(tensor, title=None):
    # copy image and do not change it.
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    plt.imshow(img)
   
    if title is not None:
        plt.title(title)
    # pause a bit to update plots.
    plt.pause(10)

"""
plt.figure()

imshow(style_img, title="Style image") 

plt.figure()
imshow(content_img, title="Content image") 
"""

class ContentLoss(nn.Module):
    def __init__(self, target):
        """Detach."""
        super(ContentLoss, self).__init__()
        self.target = target.detach()


    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        # make input transparent
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c*d) # feature per channel
    G = torch.mm(features, features.t()) # transpose
    
    return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()


    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalizaton_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalizaton_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# normalization
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)


    def forward(self, x):
        return (x - self.mean) / self.std

# a sequential model contains an ordered list of child modules.

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # just in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []
    
    # assume that cnn is a nn.Sequentail
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # the inplace doesn't work well with ContentLoss and StyleLoss, so
            # we replace it with out-of-place
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            print(i)
            print(layer)
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        
        print(name)
        print(model)
        if name in content_layers:
            print(name, "content")
            target = model(content_img).detach() # subgraph, no backward()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)
        print(model)
        if name in style_layers:
            print(name, "style")
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)
 
    # remove the redudant layers
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]

    return model, style_losses, content_losses


"""
plt.figure()
imshow(input_img, title='Input Image')
"""
# Instead of training NN, we train the input images

def get_input_optimizer(input_img):
    # this line shows that input is a parameter that requires a gradient.
    optimizer = optim.LBFGS([input_img.requires_grad_()]) # happen in place
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, 
                       content_img, style_img, input_img, num_steps=3000,
                       style_weight=10000000, content_weight=1):

    print('Building the style transfer model....')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)
    print("OPTIMIZING")

    run = [0]
    while run[0] < num_steps:
       
        # LBFGS needs to reevaluate model multiple times.   
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0.
            content_score = 0.
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1

            if run[0] % 50 == 0:
                print("run {}:".format(run)) 
                print("Style loss : {:4f} Content loss: {:5f}".format(style_score.item(), content_score.item())) 
                print() 

            return content_score + style_score

        optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)

    return input_img
            

input_img = image_loader('./images/bear.jpg')
print(input_img.size())
output = run_style_transfer(cnn, cnn_normalizaton_mean, cnn_normalizaton_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')
plt.ioff()
plt.show()
