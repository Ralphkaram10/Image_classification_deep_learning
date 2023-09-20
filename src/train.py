from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
import config.config as config_train 
from DL.classification_dl import CustomDataset 
from models.model import resnet18 as Net
from common.datakeywords import *
from common.utils import get_normalization_transform

def train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_torch_device():
    use_cuda = config_train.use_cuda and torch.cuda.is_available()
    use_mps = config_train.use_mps and torch.backends.mps.is_available()
    torch.manual_seed(config_train.seed)
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_dataloader_kwargs(batch_size=10,num_workers=1,pin_memory=True,shuffle=True):
    use_cuda = config_train.use_cuda and torch.cuda.is_available()
    kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        kwargs.update(cuda_kwargs)
    return kwargs


def main():
    
    device=get_torch_device()
    
    train_loader_kwargs=get_dataloader_kwargs(batch_size=config_train.batch_size,num_workers=1,pin_memory=True,shuffle=True)
    test_loader_kwargs=get_dataloader_kwargs(batch_size=config_train.test_batch_size,num_workers=1,pin_memory=True,shuffle=True)

    if config_train.normalize:
        transform=get_normalization_transform()
    else:
        transform=None

    dataset1=CustomDataset(config_train.train_manifest_path, transform=transform)
    dataset2=CustomDataset(config_train.test_manifest_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_loader_kwargs)

    model = Net(num_classes=config_train.num_classes).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config_train.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=config_train.gamma)
    for epoch in range(1, config_train.epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval=config_train.log_interval, dry_run=config_train.dry_run)
        test(model, device, test_loader)
        scheduler.step()

    if config_train.save_model:
        torch.save(model.state_dict(), "output/mnist_resnet.pt")


if __name__ == '__main__':
    main()
