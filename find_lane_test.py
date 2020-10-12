"""
find_lane 算法尝试
"""
import cv2
import numpy as np

from warpImg import *
from binary_threshold import *

src_img=cv2.imread('./railway_img_1280x720.jpg')
gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#逆透视变换转换为鸟瞰图
M,Min=warpImg(gray_img) #变换矩阵
warp_img=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#对原始图像进行转换
warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(src_img.shape[1],src_img.shape[0]))

# 创建ROI,这是在灰度图上创建ROI
mask = np.zeros_like(gray_img)
mask[:, 450:950] = 255
warp_grayimg_roi = cv2.bitwise_and(warp_img, warp_img, mask=mask)
warp_srcimg_roi = cv2.bitwise_and(warp_srcimg, warp_srcimg, mask=mask)

binary_img = binary_threshold(warp_grayimg_roi, warp_srcimg_roi)
cv2.imshow('binary img',binary_img)
cv2.waitKey(0)


