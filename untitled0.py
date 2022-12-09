#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:57:18 2022

@author: rishi
"""

import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
img =Image.open("dogncat1920.png")
#img = np.random.random((3, 1024, 768))
print(img.size)
conv1 = nn.Conv2d(3, 16, (5,5),stride=2,padding=1)
img=T.ToTensor()(img)
print(img.shape)
img=img.unsqueeze(0)
imgs=conv1(img)
img=imgs.squeeze(0)
print(img.size())