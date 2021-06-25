#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : main.py 
@Author : ljt
@Description: xx
@Time : 2021/6/17 19:28 
"""

from registration import *
import cv2
import numpy as np
import os
from medpy import metric #安装medpy包

img_path1 = 'data/1.tif'
img_path2 = 'data/2.tif'

orb_reg(ref_img_path=img_path1, move_img_path=img_path2, orb_num=5000, out_path="tmp.png", plt_show=True)



# 时间轴配准
# ref_img_path = "data/cat.jpg"
# dice_all = 0
# for i in range(25):
#     move_img_path = "data/cat_{}.jpg".format(i)
#     out_path = "data/cat_out/out_{}.jpg".format(i)
#     orb_reg(ref_img_path, move_img_path, orb_num=50000, out_path=out_path, plt_show=False)
#     dice = metric.dc(cv2.imread(out_path), cv2.imread(ref_img_path))
#     dice_all += dice
#     print("已完成第{}幅图像的配准！".format(i))
# print(dice_all / 24)



# orb_reg(img_path1, img_path2, orb_num=5000, out_path="dd", plt_show=True)

# gty=cv2.imread("data/cat.jpg")# 读取label文件
# gtx=cv2.imread("data/cat_out/out_19.jpg")# 读取label文件
# Hausdorff1=metric.hd(gty,gtx,voxelspacing= 0.3515625) #voxelspacing表示图像的物理距离，一般医学图像里面会有这个
# dice1=metric.dc(gty,gtx)
# #其他的评价指标metric里面有详细介绍 直接调用即可
# print(Hausdorff1)
# print(dice1)
