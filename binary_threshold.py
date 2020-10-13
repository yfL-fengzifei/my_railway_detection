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
    blur_kernel=11
    # cv2.medianBlur(warp_grayimg,blur_kernel,warp_grayimg)
    cv2.GaussianBlur(warp_grayimg,(blur_kernel,blur_kernel),0,warp_grayimg)
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
    # blur_kernel=11
    # # cv2.medianBlur(warp_grayimg,blur_kernel,warp_grayimg)
    # cv2.GaussianBlur(warp_grayimg,(blur_kernel,blur_kernel),0,warp_grayimg)

    #x\y方向梯度
    sobelx=cv2.Sobel(warp_grayimg,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(warp_grayimg,cv2.CV_64F,0,1,ksize=sobel_kernel)

    #计算幅值
    mag_sobel=np.sqrt(np.square(sobelx)+np.square(sobely))

    #转换到0-255,8-bit数据
    scaled_sobel=np.uint8(255*mag_sobel/np.max(mag_sobel))

    #创建二值图像
    binary_img=np.zeros_like(scaled_sobel)
    binary_img[(scaled_sobel>=mag_thresh[0])&(scaled_sobel<=mag_thresh[1])]=255

    return binary_img


def dir_threshold(warp_grayimg,warp_srcimg,sobel_kernel=7,dir_thresh=(0,0.09)):
    """
    sobel梯度阈值
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :param sobel_kernel: sobel核
    :param dir_thrsh: 方向阈值
    :return: 方向二值图像
    """
    # blur_kernel=11
    # # cv2.medianBlur(warp_grayimg,blur_kernel,warp_grayimg)
    # cv2.GaussianBlur(warp_grayimg,(blur_kernel,blur_kernel),0,warp_grayimg)

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
    binary_img[(dir_sobel>=dir_thresh[0])&(dir_sobel<=dir_thresh[1])]=255

    return binary_img


def magAddDir_thresh(binary_magSobelimg,binary_dirSobelimg):
    """
    综合幅度和方向
    :param binary_magSobelimg: sobel幅度
    :param binary_dirSobelimg: sobel方向
    :return: 二值图像
    """
    binary_img=np.zeros_like(binary_magSobelimg)
    binary_img[(binary_magSobelimg==1)&(binary_dirSobelimg==1)]=255

    return binary_img


def hls_select(warp_grayimg,warp_srcimg,l_thresh=(0,255)):
    """
    HLS 空间转换
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :param l_thresh: l阈值
    :return: 二值图像
    """
    blur_kernel=11
    # cv2.medianBlur(warp_srcimg,blur_kernel,warp_srcimg)
    cv2.GaussianBlur(warp_srcimg,(blur_kernel,blur_kernel),0,warp_srcimg)

    hls_img=cv2.cvtColor(warp_srcimg,cv2.COLOR_BGR2HLS)

    l_channel=hls_img[:,:,1] #范围是0-255

    # cv2.imshow('l',l_channel)
    # cv2.waitKey(0)

    #归一化，应该不用吧
    # scaled_hls=np.uint8(255*l_channel/np.max(l_channel))

    #二值化
    binary_img=np.zeros_like(l_channel)
    binary_img[(l_channel>=l_thresh[0])&(l_channel<=l_thresh[1])]=1

    return binary_img


def binary_threshold(warp_grayimg,warp_srcimg):
    """
    version1 不引入滤波 多种测试，未穷尽
    version2 初步选择 Gaussian_11 和 ab_soble 和 hls 阈值; version2中对abs引入膨胀或闭运算(暴力)
    注：原始结果是在未引入滤波的条件下的结果
    :param warp_grayimg: 灰度图像
    :param warp_srcimg: 彩色图像
    :return: 二值图像
    """
    #绝对sobel阈值
    #原始结果30-100
    binary_absSobelimg=abs_sobel_threshold(warp_grayimg,warp_srcimg,orient='x',thresh_min=100,thresh_max=190)
    # cv2.imshow('abs sobel',binary_absSobelimg)
    #膨胀或闭运算
    #结构体
    struct_kernel=np.ones((5,5),dtype=np.uint8)
    #膨胀
    # binary_absSobelimg=cv2.dilate(binary_absSobelimg,struct_kernel,iterations=3)
    #闭运算
    cv2.morphologyEx(binary_absSobelimg,cv2.MORPH_CLOSE,struct_kernel,binary_absSobelimg,iterations=3)
    # cv2.imshow('abs struct',binary_absSobelimg)
    # cv2.imwrite('./inter_result_imgs/abs_sobel_img.jpg',binary_absSobelimg)
    # cv2.waitKey(0)
    # print('pass')

    # #soble幅值阈值
    # #原始结果45-100
    # binary_magSobelimg=mag_threshold(warp_grayimg,warp_srcimg,sobel_kernel=3,mag_thresh=(100,190))
    # cv2.imshow('mag sobel',binary_magSobelimg)
    # # cv2.imwrite('./inter_result_imgs/mag_sobel_img.jpg',binary_magSobelimg)
    # # cv2.waitKey(0)
    # # print('pass')

    # #sobel方向阈值
    # #原始结果0.01-0.5
    # binary_dirSobelimg=dir_threshold(warp_grayimg,warp_srcimg,sobel_kernel=3,dir_thresh=(0.01,0.5))
    # cv2.imshow('dir sobel',binary_dirSobelimg)
    # # cv2.imwrite('./inter_result_imgs/dir_sobel_img.jpg',binary_dirSobelimg)
    # # cv2.waitKey(0)
    # # print('pass')

    #sobel幅度和方向综合阈值
    # binary_magAddDirimg=magAddDir_thresh(binary_magSobelimg,binary_dirSobelimg)
    # cv2.imshow('mag and dir',binary_magAddDirimg)
    # cv2.imwrite('./inter_result_imgs/04_mag_and_dir_img.jpg',binary_magAddDirimg)
    # cv2.waitKey(0)

    #hls l亮度阈值
    #原始结果160-255
    binary_hlsimg=hls_select(warp_grayimg,warp_srcimg,l_thresh=(130,255))
    # cv2.imshow('hls',binary_hlsimg)
    # cv2.imwrite('./inter_result_imgs/hls_img.jpg',binary_hlsimg)
    # cv2.waitKey(0)
    # print('pass')

    #下面是合并测试
    # #abs+maganddir
    # binary_absAndMagDir_img=np.zeros_like(binary_absSobelimg)
    # binary_absAndMagDir_img[(binary_absSobelimg==1)&(binary_magAddDirimg==1)]=255
    # # cv2.imshow('abd and mag_dir',binary_absAndMagDir_img)
    # # cv2.imwrite('./inter_result_imgs/06_abd_and_magdir.jpg',binary_absAndMagDir_img)
    # # cv2.waitKey(0)
    #abs+hls
    binary_absAndHls_img=np.zeros_like(binary_absSobelimg)
    binary_absAndHls_img[(binary_absSobelimg==1)&(binary_hlsimg==1)]=255
    # cv2.imshow('abs and hls',binary_absAndHls_img)
    # cv2.imwrite('./inter_result_imgs/07_abs_and_hls.jpg',binary_absAndHls_img)
    # cv2.waitKey(0)

    return binary_absAndHls_img
