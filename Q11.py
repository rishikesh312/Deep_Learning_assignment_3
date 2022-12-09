#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:39:34 2022

@author: rishi
"""
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
img_size=19
transform = transforms.Compose(
[transforms.ToTensor()])
path="mnist-varres"
training_dataset = torchvision.datasets.ImageFolder(root=path,
                                                    transform=transform)
train_load = torch.utils.data.DataLoader(training_dataset,batch_size=1,num_workers=0,shuffle=True)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_load)
images, labels = next(dataiter)
print(images.size())
#img=T.ToTensor()()
# show images
globalmax=nn.MaxPool2d(2,2)
globavg=nn.AvgPool2d(1,1)

t1=globalmax(images)
t=globavg(t1)
print(t1.size())
print(t.size())
