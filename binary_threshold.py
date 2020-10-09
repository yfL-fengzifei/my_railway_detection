import cv2
import numpy as np
import matplotlib.pyplot as plt

def abs_sobel_threshold(warp_grayimg,warp_srcimg,orient='x',thresh_min=30,thresh_max=100):
    """
    绝对梯度图
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :return: 绝对sobel二值图像
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
    binary_img[(scaled_sobel>=thresh_min)&(scaled_sobel<=thresh_max)]=1 #这里1就可以，但是在单独检测的时候可以变成255

    return binary_img


def mag_threshold(warp_grayimg,warp_srcimg,sobel_kernel=3,mag_thresh=(0,255)):
    """
    梯度方向阈值
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :param sobel_kernel: sobel核
    :param mag_thresh: 幅度阈值
    :return: 幅度图像
    """
    #x\y方向梯度
    sobelx=cv2.Sobel(warp_grayimg,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(warp_grayimg,cv2.CV_64F,0,1,ksize=sobel_kernel)

    #计算幅值
    mag_sobel=np.sqrt(np.square(sobelx)+np.square(sobely))

    #转换到0-255,8-bit数据
    scaled_sobel=np.uint8(255*mag_sobel/np.max(mag_sobel))

    #创建二值图像
    binary_img=np.zeros_like(scaled_sobel)
    binary_img[(scaled_sobel>=mag_thresh[0])&(scaled_sobel<=mag_thresh[1])]=1

    return binary_img


def dir_threshold(warp_grayimg,warp_srcimg,sobel_kernel=7,dir_thrsh=(0,0.09)):
    """
    sobel梯度阈值
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :param sobel_kernel: sobel核
    :param dir_thrsh: 方向阈值
    :return: 方向二值图像
    """
    #x\y方向梯度
    sobelx=cv2.Sobel(warp_grayimg,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(warp_grayimg,cv2.CV_64F,0,1,ksize=sobel_kernel)

    #计算x\y梯度的绝对值
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)

    #计算梯度方向
    dir_sobel=np.arctan2(abs_sobely,abs_sobelx)

    #创建二值图像
    binary_img=np.zeros_like(dir_sobel)
    binary_img[(dir_sobel>=dir_thrsh[0])&(dir_sobel<=dir_thrsh[1])]=255

    return binary_img


def binary_threshold(warp_grayimg,warp_srcimg):
    """
    :param img:
    :return:
    """
    #绝对sobel阈值
    binary_absSobelimg=abs_sobel_threshold(warp_grayimg,warp_srcimg,orient='x',thresh_min=40,thresh_max=100)
    # cv2.imshow('abs sobel',binary_absSobelimg)
    # cv2.waitKey(0)

    #soble幅值阈值
    binary_magSobelimg=mag_threshold(warp_grayimg,warp_srcimg,sobel_kernel=3,mag_thresh=(45,100))
    # cv2.imshow('mag sobel',binary_magSobelimg)
    # cv2.waitKey(0)

    binary_dirSobelimg=dir_threshold(warp_grayimg,warp_srcimg,sobel_kernel=3,dir_thresh=(0,1.3))
    cv2.imshow('dir sobel',binary_dirSobelimg)
    cv2.waitKey(0)
    # channel test,见fun_test
    pass
