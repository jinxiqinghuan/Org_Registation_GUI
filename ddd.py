#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : ddd.py 
@Author : ljt
@Description: xx
@Time : 2021/6/25 9:18 
"""



import cv2 as cv
import matplotlib.pyplot as plt



path = "data/cat.png"
img = plt.imread(path)
print(img.shape)

# for chanle in range(img.shape[2]):
#     for w in range(img.shape[1]):
#         for h in range(img.shape[0]):
#             if img[h, w, chanle] == 0:
#
