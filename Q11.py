#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:39:34 2022

@author: rishi
"""
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn

#loads the dataset of variable resolution
transform = transforms.Compose(
[transforms.ToTensor()])
path="mnist-varres"
training_dataset = torchvision.datasets.ImageFolder(root=path,
                                                    transform=transform)
train_load = torch.utils.data.DataLoader(training_dataset,batch_size=1,num_workers=0,shuffle=True)



# get some random training images
dataiter = iter(train_load)
images, labels = next(dataiter)
#prints the image resolution in tensor format
print(images.size())
#img=T.ToTensor()()


"""Still not complete"""
globalmax=nn.MaxPool2d(2,)
globavg=nn.AvgPool2d(1)

x_m1_max = torch.max(torch.max(images,dim=-1).values,dim=-1)[0]
x_m1_avg = torch.mean(torch.mean(images,dim=-1),dim=-1)

print("Gobal max pooling: ",x_m1_max.size())
print("Gobal average pooling: ",x_m1_avg.size())

