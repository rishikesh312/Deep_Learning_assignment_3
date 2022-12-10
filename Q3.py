#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:11:45 2022

@author: rishi
"""
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
  def __init__(self,in_channels, out_channels, kernel_size=(3,3),stride=1, padding=1):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
    kernel_size=kernel_size, stride=stride, padding=padding)

  def forward(self, input_batch):
    b, c, h, w = input_batch.size()
    return self.conv(input_batch)

conv = Conv2D(in_channels=3, out_channels=16)
input_batch = torch.randn(16, 3, 32, 32)
output_batch = conv(input_batch)
x1=output_batch
#print(output_batch)

class Conv2DFunc(torch.autograd.Function):
  def forward(ctx, input_batch, kernel, stride=1, padding=1):
    ctx.save_for_backward(input_batch)
    ctx.save_for_backward(kernel)
        # Apply convolution using given kernel, stride, and padding
    output_batch = F.conv2d(input_batch, kernel, stride=stride, padding=padding)
    return output_batch

  def backward(ctx, grad_output):
    input_batch, kernel = ctx.saved_tensors

    # Compute gradients for input and kernel
    input_batch_grad = F.conv2d(grad_output, kernel, stride=1, padding=1)
    kernel_grad = F.conv2d(input_batch, grad_output, stride=1, padding=1)
    #print("testting")
    return input_batch_grad, kernel_grad, None, None

input_batch = torch.randn(16, 3, 32, 32)
kernel = torch.randn(16, 3, 3, 3)
output_batch=Conv2DFunc.apply(input_batch, kernel)
x2=output_batch
for i in range(1):
    print("Automated One: \n")
    print(x1)
    print("personalized One: \n")
    print(x2)
    
