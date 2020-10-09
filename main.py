import cv2
import numpy as np
import matplotlib.pyplot as plt

from warpImg import *
from binary_threshold import *

def main(src_img):
    """
    :return:
    """
    #灰度转换
    gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

    #逆透视变换转换为鸟瞰图
    M,Min=warpImg(gray_img) #变换矩阵
    warp_img=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
    #对原始图像进行转换
    # warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(src_img.shape[1],src_img.shape[0]))
    # plt.subplot(1,2,1)
    # plt.imshow(warp_img)
    # plt.subplot(1,2,2)
    # plt.imshow(warp_srcimg)
    # plt.show()

    #创建ROI,这是在灰度图上创建ROI
    mask=np.zeros_like(gray_img)
    mask[:,450:950]=255
    warp_grayimg_roi=cv2.bitwise_and(warp_img,warp_img,mask=mask)
    warp_srcimg_roi=cv2.bitwise_and(src_img,src_img,mask=mask)
    # cv2.imshow('warp roi',warp_grayimg_roi)
    # cv2.imshow('warp src roi',warp_srcimg_roi)
    # cv2.waitKey(0)

    #二值化
    binary_threshold(warp_grayimg_roi,warp_srcimg_roi)

    pass

if __name__=="__main__":
    #检测单张图像
    src_img=cv2.imread('./railway_img_1280x720.jpg')
    # cv2.imshow('src',src_img)
    # cv2.waitKey(0)

    main(src_img)
