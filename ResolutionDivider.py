#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:06:23 2022

@author: rishi
"""


import glob
import shutil 
from PIL import Image
import os

resolution = {'32': (32, 32),'48': (48, 48),'64': (64, 64)}

for size in ['32', '48', '64']:
    #print("img")
    new_main = os.path.join('mnist-varres-' + size)
    if not os.path.exists(new_main):
        os.makedirs(new_main)

    origin = glob.glob(os.path.join('mnist-varres', '*'))
    for fld in origin:
        fld_name = os.path.normpath(fld).split(os.sep)[-1]
        new_folder = os.path.join(new_main, fld_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        sub_fld = glob.glob(os.path.join(fld, '*'))
        for main_sub_fld in sub_fld:
            sub_fld_name = os.path.normpath(main_sub_fld).split(os.sep)[-1]
            new_fld_sub = os.path.join(new_folder, sub_fld_name)
            if not os.path.exists(new_fld_sub):
                os.makedirs(new_fld_sub)

            for main_img in glob.glob(os.path.join(main_sub_fld, '*.png')):
                img = Image.open(main_img)
                desired_size = resolution[size]
                if (img.width, img.height) == desired_size:
                    img_name = os.path.normpath(main_img).split(os.sep)[-1]
                    new_img = os.path.join(new_fld_sub, img_name)
                    if not os.path.exists(new_img):
                        shutil.copy(main_img, new_img)
print("Done!")
