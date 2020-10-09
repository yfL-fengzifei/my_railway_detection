import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

#======================================================glob
# images=glob.glob('./calibration_imgs/*.jpg') #glob是一个标准库，glob.glob利用通配符返回文件的名字
# for img_path in images:
#     print(img_path)


#======================================================warpimg
# img=cv2.imread('./railway_img_1280x720.jpg',0)
#
# img_size=img.shape[:2]
# print(img_size)
#
# src_pts = np.float32([[(400, 720), (566, 200), (666, 200), (1066, 720)]])
# dst_pts = np.float32([[(450, 720), (450, 0), (950, 0), (950, 720)]])
# # cv2.polylines(img,np.int_([src_pts]),True,(255,0,0))
# # cv2.fillPoly(img,np.int_([src_pts]),(250,0,0))
# # plt.subplot(2,2,1)
# # plt.imshow(img)
# M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# Min = cv2.getPerspectiveTransform(dst_pts, src_pts)
# warp_img=cv2.warpPerspective(img,M,(img_size[1],img_size[0]))
# # cv2.polylines(img,np.int_([dst_pts]),True,(255,0,0))
# # p2=cv2.fillPoly(img,np.int_([dst_pts]),(250,0,0))
# # plt.subplot(2,2,2)
# # plt.imshow(p2)
#
#
# mask=np.zeros_like(img)
# mask[:,450:950]=255
# plt.subplot(2,2,3)
# plt.imshow(mask)
#
#
# test=cv2.bitwise_and(warp_img,warp_img,mask=mask)
# plt.subplot(2,2,4)
# plt.imshow(test)
#
# plt.show()
# # cv2.imshow('src_pts',img)
# # cv2.waitKey(0)

#======================================================channel_test
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_img=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# plt.figure(1)
# plt.subplot(2,2,1)
# plt.imshow(warp_img)
# plt.subplot(2,2,2)
# plt.imshow(warp_srcimg)

# #创建ROI
# mask=np.zeros_like(gray_img)
# mask[:,450:950]=255
# warp_img_roi=cv2.bitwise_and(warp_img,warp_img,mask=mask)
# warp_srcimg_roi=cv2.bitwise_and(warp_srcimg,warp_srcimg,mask=mask)
# plt.subplot(2,2,3)
# plt.imshow(warp_img_roi)
# plt.subplot(2,2,4)
# plt.imshow(warp_srcimg_roi)
# plt.show()
#
# r_img=warp_srcimg_roi[:,:,0]
# g_img=warp_srcimg_roi[:,:,1]
# b_img=warp_srcimg_roi[:,:,2]
# plt.figure(2)
# plt.subplot(3,3,1)
# plt.imshow(r_img)
# plt.subplot(3,3,2)
# plt.imshow(g_img)
# plt.subplot(3,3,3)
# plt.imshow(b_img)
#
#
# warp_srcimg_roi_hsv=cv2.cvtColor(warp_srcimg_roi,cv2.COLOR_RGB2HSV)
# h_img=warp_srcimg_roi_hsv[:,:,0]
# s_img=warp_srcimg_roi_hsv[:,:,1]
# v_img=warp_srcimg_roi_hsv[:,:,2]
# plt.subplot(3,3,4)
# plt.imshow(h_img)
# plt.subplot(3,3,5)
# plt.imshow(s_img)
# plt.subplot(3,3,6)
# plt.imshow(v_img)
#
#
# warp_srcimg_roi_lab=cv2.cvtColor(warp_srcimg_roi,cv2.COLOR_RGB2LAB)
# l_img=warp_srcimg_roi_lab[:,:,0]
# a_img=warp_srcimg_roi_lab[:,:,1]
# lab_img=warp_srcimg_roi_lab[:,:,2]
# plt.subplot(3,3,7)
# plt.imshow(l_img)
# plt.subplot(3,3,8)
# plt.imshow(a_img)
# plt.subplot(3,3,9)
# plt.imshow(lab_img)
# plt.show()
#
# warp_srcimg_roi_hls=cv2.cvtColor(warp_srcimg_roi,cv2.COLOR_RGB2HLS)
# hlsh_img=warp_srcimg_roi_hls[:,:,0]
# hlsl_img=warp_srcimg_roi_hls[:,:,1]
# hlss_img=warp_srcimg_roi_hls[:,:,2]
# plt.figure(3)
# plt.subplot(1,3,1)
# plt.imshow(hlsh_img)
# plt.subplot(1,3,2)
# plt.imshow(hlsl_img)
# plt.subplot(1,3,3)
# plt.imshow(hlsh_img)
# plt.show()


#======================================================threshold阈值
src_img=cv2.imread('./railway_img_1280x720.jpg')
gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

#逆透视变换转换为鸟瞰图
from warpImg import *
M,Min=warpImg(gray_img) #变换矩阵
warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))

mask = np.zeros_like(gray_img)
mask[:, 450:950] = 255
warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
warp_srcimg_roi = cv2.bitwise_and(src_img, src_img, mask=mask)

from binary_threshold import *
# binary_absSobelimg=abs_sobel_threshold(warp_grayimg_roi,warp_srcimg_roi,orient='x',thresh_min=50,thresh_max=200)
# # cv2.imshow('abs sobel',binary_absSobelimg)
# # cv2.waitKey(0)

#==滑动条测试-结果为0-100
src_img=cv2.imread('./railway_img_1280x720.jpg')
gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

def nothing(x):
    pass
cv2.namedWindow('abs sobel')
cv2.createTrackbar('min','abs sobel',0,255,nothing)
cv2.createTrackbar('max','abs sobel',0,255,nothing)
while(1):
    thresh_min=cv2.getTrackbarPos('min','abs sobel')
    thresh_max=cv2.getTrackbarPos('max','abs sobel')
    binary_absSobelimg = abs_sobel_threshold(warp_grayimg_roi, warp_srcimg_roi, orient='x', thresh_min=thresh_min,
                                             thresh_max=thresh_max)
    cv2.imshow('abs sobel',binary_absSobelimg)
    k=cv2.waitKey(1)& 0xFF
    if k==27:
        break
cv2.destroyAllWindows()

