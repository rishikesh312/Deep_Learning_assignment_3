#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:38:43 2022

@author: rishi
"""
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import math
#change the value of N. N can be 32 , 48 , 64
N=64
path="mnist-varres-"+str(N)
transform = transforms.Compose(
[transforms.ToTensor()])
training_dataset = torchvision.datasets.ImageFolder(root=path,
                                                    transform=transform)
#calculate the dataset size and divide them
dataset_size=len(training_dataset)
print(len(training_dataset))
training_size=math.ceil(0.6*dataset_size)
validation_size=math.floor(0.4*dataset_size)


train_load, valid_load = torch.utils.data.random_split(training_dataset, [training_size, validation_size])
train_load = torch.utils.data.DataLoader(training_dataset,batch_size=256,num_workers=3,shuffle=True)
val_load = torch.utils.data.DataLoader(training_dataset,batch_size=256,num_workers=3,shuffle=True)


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.con1 = nn.Conv2d(3, 16,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(16, 32,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.con3 = nn.Conv2d(32, 81, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.lin=nn.Linear(81,10)
    def forward(self,x):
        x=self.con1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)
        x=self.con2(x)
        x=self.relu2(x)
        x=self.maxpool2(x)
        x=self.con3(x)
        x=self.relu3(x)
        x=self.maxpool3(x)
        x=torch.max(torch.max(x,-1).values,-1)[0]
        x=self.lin(x)
        return x

model = NN()
loss_function=nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

#Training
for epoch in range(10):
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_load):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(f'Epochs {epoch+1} - Training Accuracy: {100 * correct // total} %')
    
    for X,y in val_load:
        y_pred = model(X)
        _,predicted = torch.max(y_pred.data,1)
    total +=y.size(0)
    correct += (predicted == y).sum().item()
    print("Epoch",epoch+1," - Validation Accuracy:",(correct/total)*100)

          
        
print('Completed Successfully')
   
