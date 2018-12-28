from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt


epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
pretrained_model = 'lenet.pt'

use_cuda=True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2d_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return F.log_softmax(x, dim=1)



test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=1, shuffle=True)

print("CUDA available: ", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

model.load_state_dict(torch.load('./lenet_mnist_model.pth'))

model.eval()

# create perturbed images
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # clip image to [0, 1]
    perturbed_image.clamp_(0, 1)

    return perturbed_image


def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)


        data.requires_grad = True
        output = model(data)
        init_pred = output.max(dim=1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        # cal loss
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        # reclassify
        perturbed_outputs = model(perturbed_data.to(device))
        perturbed_preds = torch.max(perturbed_outputs, dim=1, keepdim=True)[1]

        if perturbed_preds.item() == target.item():
            correct += 1
            if epsilon == 0 and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), perturbed_preds.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), perturbed_preds.item(), adv_ex))



    final_acc = correct / float(len(test_loader))
    print("Epsilon: {} \t Test accuracy = {} / {} = {}".format(
           epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples


accuracies = []
examples = []
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
"""
plt.figure()
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
import pickle
with open('adver.pkl', 'w') as fid:
    pickle.dump({'accuracy': accuracies, 'examples': examples}, fid)
"""

cnt = 0
plt.figure(figsize=(8,10))

for i, eps in enumerate(epsilons):
    for j, ex_info in enumerate(examples[i]):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i], fontsize=15))
        orig, adv, ex = ex_info
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap='gray')
plt.tight_layout()
plt.show()

