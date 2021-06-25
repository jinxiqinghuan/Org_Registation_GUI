#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : registration.py 
@Author : ljt
@Description: xx
@Time : 2021/6/25 9:50 
"""



from skimage import io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from medpy import metric




def orb_reg(ref_img_path, move_img_path, orb_num = 5000, out_path = None, plt_show=False):
    ref_img = io.imread(ref_img_path)
    move_img = io.imread(move_img_path)
    reg_img = np.uint8(ref_img)
    move_img = np.uint8(move_img)

    orb = cv.ORB_create(orb_num)
    kp1, des1 = orb.detectAndCompute(reg_img, None)
    kp2, des2 = orb.detectAndCompute(move_img, None)

    # def get_good_match(des1,des2):
    #  bf = cv.BFMatcher()
    #  matches = bf.knnMatch(des1, des2, k=2)
    #  good = []
    #  for m, n in matches:
    #   if m.distance < 0.75 * n.distance:
    #    good.append(m)
    #  return good,matches
    # goodMatch,matches = get_good_match(des1,des2)
    # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches[:20],None,flags=2)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 20 matches.
    img3 = cv.drawMatches(reg_img, kp1, move_img, kp2, matches[:20], None, flags=2)

    goodMatch = matches[:20]
    if len(goodMatch) >= 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold);
        # 其中H为求得的单应性矩阵矩阵
        # status则返回一个列表来表征匹配成功的特征点。
        # ptsA,ptsB为关键点
        # cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
        imgOut = cv.warpPerspective(move_img, H, (reg_img.shape[1], reg_img.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    # 叠加配准变换图与基准图
    rate = 0.5
    overlapping = cv.addWeighted(reg_img, rate, imgOut, 1 - rate, 0)
    # io.imsave('HE_2_IHC.png', overlapping)
    err = cv.absdiff(reg_img, imgOut)
    io.imsave(out_path, imgOut)
    print(metric.dc(imgOut, ref_img))
    if plt_show:
        # 显示对比
        plt.subplot(221)
        plt.title('orb')
        plt.imshow(img3)

        plt.subplot(222)
        plt.title('imgOut')
        plt.imshow(imgOut)

        plt.subplot(223)
        plt.title('overlapping')
        plt.imshow(overlapping)

        plt.subplot(224)
        plt.title('diff')
        plt.imshow(err)
        plt.show()
    return out_path