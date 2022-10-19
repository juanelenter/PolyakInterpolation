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
    
seed_everything(4568)

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


def map2simplex(mapping = "softmax"):
    
    assert mapping in ["Softmax", "Taylor", "NormalizedRelu", "TaylorInter", "SparseMax"], "Mapping not available."
    
    if mapping == "Softmax":
        return nn.LogSoftmax()
    elif mapping == "NormalizedRelu":
        return NormalizedRelu()
    elif mapping == "Taylor":
        return Taylor()
    elif mapping == "TaylorInter":
        return TaylorInter()
    elif mapping == "SparseMax":
        return Sparsemax()
    
class NormalizedRelu(nn.Module):
    """
    Normalized ReLU map to the simplex: x_i -> log( max{0, x_i / ||x||} )
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        norms = torch.linalg.norm(x, dim = 1)
        out = torch.log( nn.ReLU()( x*(1/norms[:, None]) ) + 1e-12)
        return out
    
class Taylor(nn.Module):
    """
    2nd order Taylor map to the simplex.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        numerator = 1 + x + 0.5*x**2
        denominator = torch.sum(numerator, dim = 1)
        out = torch.log( numerator*(1/denominator[:, None]) + 1e-12 )
        return out

class TaylorInter(nn.Module):
    """
    2nd order Taylor map to the simplex.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        numerator = 0.5 + x + 0.5*x**2
        denominator = torch.sum(numerator, dim = 1)
        out = torch.log( numerator*(1/denominator[:, None]) + 1e-12 )
        return out

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input 


class CIFARCNN(nn.Module):
   
    def __init__(self, mapping):
        
        super(CIFARCNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        
        self.map2simplex = map2simplex(mapping)


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)
        
        # map to simlex
        out = self.map2simplex(x)

        return out, x

cnn = resnet.ResNet18(mapping="TaylorInter")
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(10):
    train(epoch)
    test(epoch)
    scheduler.step()

def evaluate(cnn, loader):
    cnn.eval()
    with torch.no_grad():
        loss = []
        acc = []
        for images, labels in loader:
            output, last_layer = cnn(images)
            loss.append(loss_func(output, labels))
            pred_y = torch.max(output, 1)[1].data.squeeze()
            acc.append( (pred_y == labels).sum().item() / float(labels.size(0)) )       
        return acc, loss

def train(num_epochs, cnn, loaders):
    
    show_every = 150
    
    total_step = len(loaders['train'])
    
    train_metrics = {'loss' : [], 'acc' : []}
    test_metrics = {'loss' : [], 'acc' : []}
    
    for epoch in range(num_epochs):
        cnn.train()
        for i, (images, labels) in enumerate(loaders['train']):
            
            b_x = Variable(images)   
            b_y = Variable(labels)   
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            optimizer.zero_grad()           
            loss.backward()  
            optimizer.step()                
            
            if (i+1) % show_every == 0:
                pred_y = torch.max(output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item()/len(labels)
                train_metrics['loss'].append(loss.item())
                train_metrics['acc'].append(accuracy)
                print(f"Epoch: {epoch+1:.2f} , Loss: {loss.item():.2f} , Accuracy: {accuracy:.2f} ")
        
        # Gather test metrics
        accuracy, loss = evaluate(cnn, loaders['test'])
        test_metrics['loss'].append(np.mean(loss))
        test_metrics['acc'].append(np.mean(accuracy))
        
        # Decrease LR
        scheduler.step()
        
    return train_metrics, test_metrics

def train_sps(num_epochs, cnn, loaders):
    
    # Train the model
    total_step = len(loaders['train'])
    train_metrics = {'loss' : [], 'acc' : []}
    test_metrics = {'loss' : [], 'acc' : []}
    for epoch in range(num_epochs):
        cnn.train()
        for i, (images, labels) in enumerate(loaders['train']):
            
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            
            def closure():
                loss = loss_func(output, b_y)
                loss.backward()
                return loss
            
            optimizer.zero_grad()
            optimizer.step(closure = closure)  
                        
            if (i+1) % 150 == 0:
                pred_y = torch.max(output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item()
                train_metrics['loss'].append(loss.item())
                train_metrics['acc'].append(accuracy)
                print(f"Epoch: {epoch+1:.2f} , Loss: {loss.item():.2f} , Accuracy: {accuracy/len(labels):.2f} ")

        # Gather test metrics
        accuracy, loss = evaluate(cnn, loaders['test'])
        test_metrics['loss'].append(np.mean(loss))
        test_metrics['acc'].append(np.mean(accuracy))

        
    return train_metrics, test_metrics

cnn = CIFARCNN("TaylorInter")
loss_func = nn.NLLLoss()
optimizer = optim.SGD(cnn.parameters(), lr = 0.0005, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
print(cnn)
print(f"Number of Parameters = {sum(p.numel() for p in cnn.parameters() if p.requires_grad)}")
print(f"Dimensionality of the input = {32*32*3}")

num_epochs = 30
train_metrics, test_metrics = train(num_epochs, cnn, loaders)
    
np.save('./output/taylor_inter/evol_train_loss.npy', train_metrics['loss'])
np.save('./output/taylor_inter/evol_train_acc.npy', train_metrics['acc'])
np.save('./output/taylor_inter/evol_test_loss.npy', test_metrics['loss'])
np.save('./output/taylor_inter/evol_test_acc.npy', test_metrics['acc'])

# Compute final Loss in Train
acc_tr, loss_tr = evaluate(cnn, loaders["train"])
# Compute final Loss in Test
acc_te, loss_te = evaluate(cnn, loaders["test"])

np.save('./output/taylor_inter/final_test_loss.npy', loss_te)
np.save('./output/taylor_inter/final_test_acc.npy', acc_te)
np.save('./output/taylor_inter/final_train_loss.npy', loss_tr)
np.save('./output/taylor_inter/final_train_acc.npy', acc_tr)
