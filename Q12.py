#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:36:20 2022

@author: rishi
"""

import torch
import torchvision
import torch.nn as nn
import math
from torchvision import transforms

resize=28
transform = transforms.Compose(
[transforms.ToTensor(),transforms.Resize(resize)])
path="mnist-varres-64"
training_dataset = torchvision.datasets.ImageFolder(root=path,
                                                    transform=transform)
dataset_size=len(training_dataset)
print(len(training_dataset))
training_size=math.ceil(0.6*dataset_size)
validation_size=math.floor(0.4*dataset_size)


train_dl, valid_dl = torch.utils.data.random_split(training_dataset, [training_size, validation_size])
trainloader = torch.utils.data.DataLoader(train_dl, batch_size=256, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valid_dl, batch_size=256, shuffle=True, num_workers=2)



class CnnP2(nn.Module):
    """
    Convolutional Neural Network Architecture based on the description. 
    * Layers initialization.
    * Forward() function.
    """

    def __init__(self):
        super(CnnP2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        self.linear = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

# Calling Train set.
model_base = CnnP2()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_base.parameters(), lr=0.001)

print("Number of parameters: ", sum(p.numel() for p in model_base.parameters()))

for epoch in range(10):  # loop over the dataset multiple times

    correct = 0.0
    total = 0.0

    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data

        # forward + backward + optimize
        #
        outputs = model_base(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #print("Hello")
    correct += (predicted == labels).sum().item()
    print(f'Iter {epoch+1} - Training Accuracy: {100 * correct // total} %')
    for x, y in valloader:
      y_pred = model_base(x)

      _, predicted = torch.max(y_pred.data, 1)

      total += y.size(0)
      correct += (predicted == y).sum().item()

    print(f'Iter {epoch+1} - Validation Accuracy: {100 * (correct / total)} %')

print('Finished Training!')
