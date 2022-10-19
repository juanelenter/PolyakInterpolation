# Libraries
import random
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import resnet

from sps import Sps, SpsL1, SpsL2, ALIG

import wandb

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    wandb.log({"epoch": epoch, "train_acc": 100.*correct/total, "train_loss": train_loss/(batch_idx+1)})


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        wandb.log({"epoch": epoch, "test_acc": 100.*correct/total, "test_loss": test_loss/(batch_idx+1)})

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Simplex Map')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mapping', default = "Softmax", type=str)
parser.add_argument('--name', default = "NoName", type=str)
args = parser.parse_args()

wandb.init(project="PolyakInterpolation", entity="elenter", name = args.name)
wandb.log({'data': opts.data, 'seed': opts.seed, 'epsilon': opts.epsilon, 'lr':opts.lr, 'nPat': opts.nPat, 'dlr': opts.dlr, 'redund': opts.redund})

seed_everything(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Data setup
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model setup
net = resnet.ResNet18(mapping=args.mapping)
print(net)
print(f"Number of Parameters = {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
print(f"Dimensionality of the input = {32*32*3}")

# Training setup
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(10):
    train(epoch)
    test(epoch)
    scheduler.step()

