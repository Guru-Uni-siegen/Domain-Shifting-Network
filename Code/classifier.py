"""
"A Simple Domain Shifting Network for Generating Low Quality Images" implementation

Step 4 - Classifier
"""
import os
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


file_separator = "\\"

# split_train_path must be directed to the newly generated images in Step 3
split_train_path = file_separator.join(['..','Dataset','original_imagenet_images','train'])
valid_path =  file_separator.join(['..','Dataset','original_imagenet_images','val'])
batch_size = 32
epochs = 1
lr = 1e-4

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.__dict__["mobilenet_v2"](pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier[1] = nn.Linear(1280, 5)
        
        for param in self.model.features[14:].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

def train(train_loader, model, criterion, optimizer, epoch,scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    
    # switch to train mode
    model.train()
    scheduler.step()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data, images.size(0))
        acc.update(prec1, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print('   * EPOCH {epoch} | Train Accuracy: {acc.avg:.3f} | Train Loss: {losses.avg:.3f}'.format(epoch=epoch, acc=acc, losses=losses),end = ' ')
    return acc.avg
    
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(labels)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data, images.size(0))
        acc.update(prec1, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' | Val Accuracy: {acc.avg:.3f} | Val Loss: {losses.avg:.3f}'.format(acc=acc, losses=losses))

    return acc.avg
  
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = lr * (0.1**(epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
    
def main(mode="train"):
    model = Network()

    traindir = split_train_path
    valdir = valid_path

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_loader = data.DataLoader(
        datasets.ImageFolder(traindir,
                             transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)


    val_loader = data.DataLoader(
        datasets.ImageFolder(valdir,
                             transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    
    if mode == "validate":
        validate(val_loader, model, criterion, 0)
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3,step_size_up=20,mode='exp_range')
    for epoch in range(epochs):
        prec0 = train(train_loader, model, criterion, optimizer, epoch,scheduler)

        # Evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
        },epoch)

# Specify the directory for saving the trained models
def save_checkpoint(state, epoch,filename='checkpoint.pth.tar'):
    torch.save(state, file_separator.join(['..','Models','epoch_'+str(epoch)+'_'+filename]))
    
if __name__ == '__main__':              
    main(mode="train")