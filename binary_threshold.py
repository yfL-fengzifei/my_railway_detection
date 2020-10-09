import cv2
import numpy as np
import matplotlib.pyplot as plt

def abs_sobel_threshold(warp_grayimg,warp_srcimg,orient='x',thresh_min=30,thresh_max=100):
    """
    绝对梯度图
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :return:
    """
    #对灰度图像进行转换
    if orient=='x':
        sobel_img=cv2.Sobel(warp_grayimg,cv2.CV_64F,1,0)
    if orient=='y':
        sobel_img=cv2.Sobel(warp_grayimg,cv2.CV_64F,0,1)
    abs_sobel=np.absolute(sobel_img)

    #归一化到0-255,并转换为8-bit数据
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    # cv2.imshow('scaled sobel',scaled_sobel)
    # cv2.waitKey(0)
    # print('scale img size',scaled_sobel.shape)

    #创建二值图像
    binary_img=np.zeros_like(scaled_sobel)
    binary_img[(scaled_sobel>=thresh_min)&(scaled_sobel<=thresh_max)]=255 #这里1就可以，但是在单独检测的时候可以变成255

    return binary_img


def binary_threshold(warp_grayimg,warp_srcimg):
    """
    :param img:
    :return:
    """
    #channel test,见fun_test
    binary_absSobelimg=abs_sobel_threshold(warp_grayimg,warp_srcimg,orient='x',thresh_min=30,thresh_max=100)
    cv2.imshow('abs sobel',binary_absSobelimg)
    cv2.waitKey(0)

    pass
