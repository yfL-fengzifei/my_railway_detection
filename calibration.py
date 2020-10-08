"""
not completed
"""

import cv2
import numpy as np
import glob

images=glob.glob('./calibration_imgs/*.jpg') #glob是一个标准库，glob.glob利用通配符返回文件的名字,返回的是一个List

for i,img_path in enumerate(images):
    img=cv2.imread(img_path) #读图
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图

    #检查棋盘角点
    cv2.findChessboardCorners(gray_img,(9,6),None)

print('pass')
