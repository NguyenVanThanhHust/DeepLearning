import torch
import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

import torch
import torchvision.ops
from torch import nn

from dcn import DeformableConv2d

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch
from torch import nn

class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):

        super(MNISTClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)   
        conv = nn.Conv2d if deformable==False else DeformableConv2d
        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x) # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x) # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def train_one_epoch(model, dataloader, loss_func, optimizer, device):
    model.train()
    for batch_idx, (ims, tgts) in enumerate(dataloader):
        ims, tgts = ims.to(device), tgts.to(device)
        optimizer.zero_grad()
        outputs = model(ims)
        loss = loss_func(outputs, tgts)
        loss.backward()
        optimizer.step()

def test_one_epoch(model, dataloader, loss_func, device):
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0
    with torch.no_grad():
        for batch_idx, (ims, tgts) in enumerate(dataloader):
            ims, tgts = ims.to(device), tgts.to(device)
            for scale in np.arange(0.5, 1.6, 0.1): # [0.5, 0.6, ... ,1.2, 1.3, 1.4, 1.5]
                ims = transforms.functional.affine(ims, scale=scale, angle=0, translate=[0,0],shear=0)
                outputs = model(ims)
                test_loss += loss_func(outputs, tgts).item()  # sum up batch mean loss
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(tgts.view_as(pred)).sum().item()
                num_data += len(ims)

    test_loss /= num_data

    test_acc = 100. * correct / num_data
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, num_data,
        test_acc))
    return test_acc

def main(use_deformable_conv=False):
    # Training settings
    seed=1
    setup_seed(seed)

    use_cuda = torch.cuda.is_available()
    batch_size = 128
    lr=1e-3
    gamma=0.7
    epochs=40

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=train_transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MNISTClassifier(use_deformable_conv).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    loss_func = nn.CrossEntropyLoss()
    best_test_acc = 0.
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, loss_func, optimizer, device)
        best_test_acc = max(best_test_acc, test_one_epoch(model, test_loader, loss_func, device))
        scheduler.step()
    print("best top1 acc(%): ", f"{best_test_acc:.2f}")


# main(use_deformable_conv=False)

# main(use_deformable_conv=True)