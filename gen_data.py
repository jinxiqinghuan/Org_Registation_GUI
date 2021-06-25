#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : gen_data.py 
@Author : ljt
@Description: xx
@Time : 2021/6/17 18:49 
"""
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk

# 2D
path = "data/cat.jpg"

# 转换原始图像为单通道
img = Image.open(path).convert("L")
img = np.array(img)
cv.imwrite(path, img)

# 反色处理
fan = False
if fan:
    tmp_img = np.zeros((img.shape[0], img.shape[1]))
    tmp_img[img == 0] = 255
    tmp_img[img == 255] = 0
    cv.imwrite(path, tmp_img)



# 移动图像
# n = 0
# new_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
# p = -180
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if img[i][j] == 0:
#             new_img[i + p - 55][j + p - 20] = 255
# cv.imwrite("data/cat_{}.jpg".format(n), new_img)
# print("正在生成位置{}的图片".format(n))
# n += 1
# new_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

# 移动图像
# 指定位置进行平移
move = False
if move:
    n=0
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for p in range(0, 500, 20):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    new_img[i + p][j + p] = 255
        cv.imwrite("data/cat_{}.jpg".format(n), new_img)
        print("正在生成位置{}的图片".format(n))
        n += 1
        new_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")



# 图像随机生成函数

# def gen_img(img_path, resize=False, ):
#     img = cv.imread(img_path)
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     if resize:
#         size = (random.randint(1, img.shape[0]),  random.randint(1, img.shape[1]))
#         cv.resize(img, size)













