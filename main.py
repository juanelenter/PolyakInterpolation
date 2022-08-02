# Libraries
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets import get_dataset
from sps import Sps, SpsL1, SpsL2, ALIG

# Reproducibility
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_all_seeds(4568)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

dataset_name = "CIFAR"
train_data, test_data, loaders = get_dataset(dataset_name)

def map2simplex(mapping = "softmax"):
    
    assert mapping in ["Softmax", "Taylor", "NormalizedRelu", "TaylorInter"], "Mapping not available."
    
    if mapping == "Softmax":
        return nn.LogSoftmax()
    elif mapping == "NormalizedRelu":
        return NormalizedRelu()
    elif mapping == "Taylor":
        return Taylor()
    elif mapping == "TaylorInter":
        return TaylorInter()
    
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
