#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : gui_run.py 
@Author : ljt
@Description: xx
@Time : 2021/6/25 11:48 
"""
# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QWidget, QPushButton, QFileDialog, QLineEdit
from PyQt5.Qt import *
import os
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


class gui(QWidget):
    def __init__(self):
        super(gui, self).__init__()
        self.ref_img_path = None
        self.move_img_path = None
        self.l_move_img_path = None
        self.obr_num = 5000

        self.resize(1000, 600)
        self.setWindowTitle("时间轴配准2D_Version")

        self.label_ref = QLabel(self)
        self.label_ref.setText("参考图像")
        self.label_ref.setFixedSize(300, 200)
        self.label_ref.move(10, 100)

        self.label_ref.setStyleSheet("QLabel{background:white;}"
                                     "QLabel{color:rgb(300,300,300,120);font-size:16px;font-weight:bold;font-family:宋体;}"
                                     )

        self.label_move = QLabel(self)
        self.label_move.setText("移动图像")
        self.label_move.setFixedSize(300, 200)
        self.label_move.move(330, 100)
        self.label_move.setStyleSheet("QLabel{background:white;}"
                                      "QLabel{color:rgb(300,300,300,120);font-size:16px;font-weight:bold;font-family:宋体;}"
                                      )

        self.label_out = QLabel(self)
        self.label_out.setText("输出图像")
        self.label_out.setFixedSize(300, 200)
        self.label_out.move(650, 100)

        self.label_out.setStyleSheet("QLabel{background:white;}"
                                     "QLabel{color:rgb(300,300,300,120);font-size:16px;font-weight:bold;font-family:宋体;}"
                                     )

        self.label_info = QLabel(self)
        self.label_info.setText("Info:")
        self.label_info.setFixedSize(900, 50)
        self.label_info.move(50, 400)

        self.label_info.setStyleSheet("QLabel{background:white;}"
                                     "QLabel{color:rgb(300,300,300,120);font-size:16px;font-weight:bold;font-family:宋体;}"
                                     )





        # self.label2 = QLabel(self)
        # self.label2.setText("显示信息"
        #                     + "\n" + "\n"
        #                     + "注意：" + "\n"
        #                     + "路径和图片文件名不能包含中文~~~"
        #                     )
        # self.label2.setFixedSize(400, 200)
        # self.label2.move(60, 200)
        # self.label2.setStyleSheet("QLabel{background:white;}"
        #                          "QLabel{color:rgb(300,300,300,120);font-size:16px;font-weight:bold;font-family:宋体;}"
        #                          )

        btn1 = QPushButton(self)
        btn1.setText("参考图像")
        btn1.move(10, 30)
        btn1.clicked.connect(self.open_ref_image)

        btn2 = QPushButton(self)
        btn2.setText("移动图像")
        btn2.move(100, 30)
        btn2.clicked.connect(self.open_move_image)

        btn3 = QPushButton(self)
        btn3.setText("L_移动图像")
        btn3.move(190, 30)
        btn3.clicked.connect(self.open_move_path)

        btn4 = QPushButton(self)
        btn4.setText("开始配准")
        btn4.move(280, 30)
        btn4.clicked.connect(self.ref)

        btn5 = QPushButton(self)
        btn5.setText("时间轴配准")
        btn5.move(370, 30)
        btn5.clicked.connect(self.Longitude_Registration)

        btn6 = QPushButton(self)
        btn6.setText("obr_num")
        btn6.move(460, 30)
        btn6.clicked.connect(self.input_obr_num)

    def open_ref_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开参考图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label_ref.width(), self.label_ref.height())
        self.label_ref.setPixmap(jpg)
        # image_ref = cv.imread(imgName, cv.IMREAD_GRAYSCALE)
        self.ref_img_path = imgName
        self.label_info.setText("已选择参考图片:{}".format(imgName))

    def open_move_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开移动图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label_move.width(), self.label_move.height())
        self.label_move.setPixmap(jpg)
        # image_move = cv.imread(imgName, cv.IMREAD_GRAYSCALE)
        self.move_img_path = imgName
        self.label_info.setText("已选择移动图片:{}".format(imgName))


    def open_move_path(self):
        imgpath = QFileDialog.getExistingDirectory(self, "打开移动图像路径", "")
        self.l_move_img_path = imgpath
        self.label_move.setText("已选定路径！")
        self.label_info.setText("已选择时间轴配准路径:{}".format(imgpath))
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        # jpg = QtGui.QPixmap(imgName).scaled(self.label_move.width(), self.label_move.height())
        # self.label_move.setPixmap(jpg)
        # # image_move = cv.imread(imgName, cv.IMREAD_GRAYSCALE)
        # self.move_img_path = imgName
        # print(self.move_img_path)
        #

    def ref(self):
        out_path = orb_reg(ref_img_path=self.ref_img_path, move_img_path=self.move_img_path, orb_num=self.obr_num,
                           out_path="out.png", plt_show=True)
        jpg = QtGui.QPixmap(out_path).scaled(self.label_out.width(), self.label_out.height())
        self.label_out.setPixmap(jpg)


    def Longitude_Registration(self):
        l_move_img_path = self.l_move_img_path
        ref_img_path = self.ref_img_path
        fileList = sorted(os.listdir(l_move_img_path))
        dice_all = 0
        # print(fileList)
        for i in range(len(fileList)):
            move_img_path = "{}/{}".format(l_move_img_path, fileList[i])
            out_path = "out/out_{}.png".format(i)
            orb_reg(ref_img_path, move_img_path, orb_num=self.obr_num, out_path=out_path, plt_show=False)
            dice = metric.dc(cv.imread(out_path), cv.imread(ref_img_path))
            dice_all += dice
            self.label_info.setText("完成第一张图片，Dice为：{}".format(dice))
        self.label_info.setText("完成共{}幅图像的配准，平均Dice为:{}".format(len(fileList), dice_all / len(fileList)))


    def input_obr_num(self):
        self.label_info.setText("请输入新的特征点值obr_num, 默认为5000")
        obr_num, _ = QInputDialog.getInt(self, "ddd", "dddd")
        self.obr_num = obr_num
        self.label_info.setText("已设置obr_num为：{}".format(obr_num))



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = gui()
    my.show()
    sys.exit(app.exec_())
