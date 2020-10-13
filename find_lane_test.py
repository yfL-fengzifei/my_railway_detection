"""
find_lane 算法尝试
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from warpImg import *
from binary_threshold import *


def find_lane(img):
    """
    应用滑窗算法，进行车道线拟合
    :param img:
    :return:
    """
    #对图像得到下半部分进行直方图计算，即计算每列的像素数
    histogram=np.sum(img[img.shape[0]//2:,:],axis=0)

    #下面是对直方图的显示，对算法没有帮助
    #**==**需要进一步分析**==**
    # plt.figure(figsize=(img.shape[0]/100,img.shape[1]/100),dpi=100)
    plt.plot(histogram)
    plt.xlim([0,img.shape[1]])
    # plt.axis('off')
    # plt.savefig('./hist_img.png',bbox_inches='tight')
    plt.show()

    #找到左右峰值
    #中点
    midpoint=np.int(histogram.shape[0]//2)
    # print(midpoint)
    # cv2.line(img,(midpoint,img.shape[0]),(midpoint,0),(255,255,0))
    # cv2.imshow('mid line',img)
    # cv2.waitKey(0)
    #四分之一点
    quarter_point=np.int(midpoint//2)
    #左峰值点
    # lefx_base=np.argmax(histogram[:midpoint])
    lefx_base=np.argmax(histogram[quarter_point:midpoint])+quarter_point
    # print('left peak point',lefx_base)
    #右峰值点
    # rightx_base=np.argmax(histogram[midpoint:])
    rightx_base=np.argmax(histogram[midpoint:(midpoint+quarter_point)])+midpoint
    # print('right peak point',rightx_base)

    #选择滑动窗口的数量
    nwindows=10



    pass


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
warp_grayimg_roi_copy=warp_grayimg_roi[:,450:950]
# cv2.imshow('cpoy',warp_grayimg_roi_copy)
# cv2.waitKey(0)
warp_srcimg_roi_copy=warp_srcimg_roi[:,450:950,:]
# cv2.imshow('copy',warp_srcimg_roi_copy)
# cv2.waitKey(0)

# binary_img = binary_threshold(warp_grayimg_roi, warp_srcimg_roi)
binary_img = binary_threshold(warp_grayimg_roi_copy, warp_srcimg_roi_copy)

find_lane(binary_img)

pass


