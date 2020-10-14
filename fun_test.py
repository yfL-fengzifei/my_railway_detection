import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as Image
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
#
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
# print(np.max(hlsl_img))


#======================================================threshold阈值-abs阈值,结果为0-100
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(src_img, src_img, mask=mask)
#
# from binary_threshold import *
# # binary_absSobelimg=abs_sobel_threshold(warp_grayimg_roi,warp_srcimg_roi,orient='x',thresh_min=50,thresh_max=200)
# # # cv2.imshow('abs sobel',binary_absSobelimg)
# # # cv2.waitKey(0)
#
# #==滑动条测试-结果为0-100
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# def nothing(x):
#     pass
# cv2.namedWindow('abs sobel')
# cv2.createTrackbar('min','abs sobel',0,255,nothing)
# cv2.createTrackbar('max','abs sobel',0,255,nothing)
# while(1):
#     thresh_min=cv2.getTrackbarPos('min','abs sobel')
#     thresh_max=cv2.getTrackbarPos('max','abs sobel')
#     binary_absSobelimg = abs_sobel_threshold(warp_grayimg_roi, warp_srcimg_roi, orient='x', thresh_min=thresh_min,
#                                              thresh_max=thresh_max)
#     cv2.imshow('abs sobel',binary_absSobelimg)
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()

#======================================================threshold阈值-mag阈值,结果45-100
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(src_img, src_img, mask=mask)
#
# from binary_threshold import *
# # binary_magsobelimg = mag_threshold(warp_grayimg_roi, warp_srcimg_roi, sobel_kernel=3, mag_thresh=(30, 170))
# # cv2.imshow('mag sobel',binary_magsobelimg)
# # cv2.waitKey(0)
#
# def nothing(x):
#     pass
# cv2.namedWindow('mag sobel')
# cv2.createTrackbar('min','mag sobel',0,255,nothing)
# cv2.createTrackbar('max','mag sobel',0,255,nothing)
# cv2.createTrackbar('kernel size','mag sobel',3,30,nothing)
# while(1):
#     thresh_min=cv2.getTrackbarPos('min','mag sobel')
#     thresh_max=cv2.getTrackbarPos('max','mag sobel')
#     sobel_kernel=cv2.getTrackbarPos('kernel size','mag soble')
#     binary_magSobelimg=mag_threshold(warp_grayimg_roi,warp_srcimg_roi,sobel_kernel=sobel_kernel,mag_thresh=(thresh_min,thresh_max))
#     cv2.imshow('mag sobel',binary_magSobelimg)
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()

#======================================================threshold阈值-dir阈值,结果0.01-0.5
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(src_img, src_img, mask=mask)
#
# from binary_threshold import *
# # binary_magsobelimg = mag_threshold(warp_grayimg_roi, warp_srcimg_roi, sobel_kernel=3, mag_thresh=(30, 170))
# # cv2.imshow('mag sobel',binary_magsobelimg)
# # cv2.waitKey(0)
#
# def nothing(x):
#     pass
# cv2.namedWindow('dir sobel')
# cv2.createTrackbar('min','dir sobel',0,100,nothing)
# cv2.createTrackbar('max','dir sobel',0,200,nothing)
# cv2.createTrackbar('kernel size','dir sobel',3,30,nothing)
# while(1):
#     thresh_min=cv2.getTrackbarPos('min','dir sobel')
#     thresh_max=cv2.getTrackbarPos('max','dir sobel')
#     sobel_kernel=cv2.getTrackbarPos('kernel size','dir sobel')
#     binary_dirSobelimg=dir_threshold(warp_grayimg_roi,warp_srcimg_roi,sobel_kernel=sobel_kernel,dir_thresh=(thresh_min/100,thresh_max/100))
#     cv2.imshow('dir sobel',binary_dirSobelimg)
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()

#======================================================threshold阈值-mag and dir阈值,结果基本上是上述综合的最优解
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(src_img, src_img, mask=mask)
#
# from binary_threshold import *
# def nothing(x):
#     pass
# cv2.namedWindow('sobel')
# cv2.createTrackbar('mag_min','sobel',0,255,nothing)
# cv2.createTrackbar('mag_max','sobel',0,255,nothing)
# cv2.createTrackbar('dir_min','sobel',0,100,nothing)
# cv2.createTrackbar('dir_max','sobel',0,200,nothing)
#
# while(1):
#     mag_min=cv2.getTrackbarPos('mag_min','sobel')
#     mag_max=cv2.getTrackbarPos('mag_max','sobel')
#     dir_min=cv2.getTrackbarPos('dir_min','sobel')
#     dir_max=cv2.getTrackbarPos('dir_max','sobel')
#
#     binary_magSobelimg = mag_threshold(warp_grayimg_roi, warp_srcimg_roi, sobel_kernel=3,mag_thresh=(mag_min, mag_max))
#
#     binary_dirSobelimg=dir_threshold(warp_grayimg_roi,warp_srcimg_roi,sobel_kernel=3,dir_thresh=(dir_min/100,dir_max/100))
#
#     binary_sobelimg=magAddDir_thresh(binary_magSobelimg,binary_dirSobelimg)
#
#     cv2.imshow('sobel',binary_sobelimg)
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()

#======================================================threshold阈值-hls阈值,结果160-255
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(warp_srcimg, warp_srcimg, mask=mask)
#
# from binary_threshold import *
# def nothing(x):
#     pass
# cv2.namedWindow('hls')
# cv2.createTrackbar('min','hls',0,255,nothing)
# cv2.createTrackbar('max','hls',0,255,nothing)
# while(1):
#     min=cv2.getTrackbarPos('min','hls')
#     max=cv2.getTrackbarPos('max','hls')
#
#     binary_hlsimg=hls_select(warp_srcimg_roi,warp_srcimg_roi,l_thresh=(min,max))
#     cv2.imshow('hls',binary_hlsimg)
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()

# #==小测试
# import glob
# img_name=glob.glob('./inter_result_imgs/*.jpg')
# for i,img_path in enumerate(img_name):
#     print(img_path)
#     img=cv2.imread(img_path)
#     plt.subplot(1,5,i+1)
#     plt.imshow(img)
# plt.show()

#======================================================threshold阈值-abs+hls
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(warp_srcimg, warp_srcimg, mask=mask)
#
# from binary_threshold import *
# def nothing(x):
#     pass
# cv2.namedWindow('dst')
# cv2.createTrackbar('min','dst',0,255,nothing)
# cv2.createTrackbar('max','dst',0,255,nothing)
# cv2.createTrackbar('min2','dst',0,255,nothing)
# cv2.createTrackbar('max2','dst',0,255,nothing)
# while(1):
#     thresh_min=cv2.getTrackbarPos('min','dst')
#     thresh_max=cv2.getTrackbarPos('max','dst')
#     min2=cv2.getTrackbarPos('min2','dst')
#     max2=cv2.getTrackbarPos('max2','dst')
#
#     binary_absSobelimg = abs_sobel_threshold(warp_grayimg_roi, warp_srcimg_roi, orient='x', thresh_min=thresh_min,
#                                              thresh_max=thresh_max)
#     binary_hlsimg=hls_select(warp_srcimg_roi,warp_srcimg_roi,l_thresh=(min2,max2))
#
#     binary_absAndHls_img=np.zeros_like(binary_absSobelimg)
#     binary_absAndHls_img[(binary_absSobelimg==1)&(binary_hlsimg==1)]=255
#     cv2.imshow('dst',binary_absAndHls_img)
#
#     k=cv2.waitKey(1)& 0xFF
#     if k==27:
#         break
# cv2.destroyAllWindows()
#======================================================threshold阈值-还差meganddir+hls

#======================================================================================**阈值测试
#======================================================================================**引入滤波
#===============================================================================abs=kernel_size=3,130-255
#===============================================================================**经测试创建滑动条没用**
# src_img=cv2.imread('./railway_img_1280x720.jpg')
# gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# #逆透视变换转换为鸟瞰图
# from warpImg import *
# M,Min=warpImg(gray_img) #变换矩阵
# warp_grayimg=cv2.warpPerspective(gray_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
# warp_srcimg=cv2.warpPerspective(src_img,M,dsize=(gray_img.shape[1],gray_img.shape[0]))
#
# mask = np.zeros_like(gray_img)
# mask[:, 450:950] = 255
# warp_grayimg_roi = cv2.bitwise_and(warp_grayimg, warp_grayimg, mask=mask)
# warp_srcimg_roi = cv2.bitwise_and(warp_srcimg, warp_srcimg, mask=mask)
# # cv2.imshow('warp gray roi',warp_grayimg_roi)
# # cv2.imshow('warp src roi',warp_srcimg_roi)
# # cv2.waitKey(0)
# # kernel_size=(21,21)
# # cv2.blur(warp_grayimg_roi,ksize=3,dst=warp_grayimg_roi)
# # cv2.imshow('blur',warp_grayimg_roi)
# # cv2.medianBlur(warp_grayimg_roi,ksize=11,dst=warp_grayimg_roi)
# # cv2.imshow('blur',warp_grayimg_roi)
# # # cv2.GaussianBlur(warp_grayimg_roi,kernel_size,0,warp_grayimg_roi)
# # # cv2.imshow('blur',warp_grayimg_roi)
# # # cv2.bilateralFilter(cv2.cvtColor(warp_grayimg_roi,cv2.COLOR_BGRA2BGR),9,175,175,dst=warp_grayimg_roi)
# # # cv2.imshow('blur',warp_grayimg_roi)
# # cv2.waitKey(0)
#
#
# from binary_threshold import *
# # binary_absSobelimg=abs_sobel_threshold(warp_grayimg_roi,warp_srcimg_roi,orient='x',thresh_min=50,thresh_max=200)
# # # cv2.imshow('abs sobel',binary_absSobelimg)
# # # cv2.waitKey(0)
#
# #==滑动条测试-结果为0-100
# # src_img=cv2.imread('./railway_img_1280x720.jpg')
# # gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
#
# def nothing(x):
#     pass
# cv2.namedWindow('abs sobel')
# # cv2.createTrackbar('min','abs sobel',0,255,nothing)
# # cv2.createTrackbar('max','abs sobel',0,255,nothing)
# # cv2.createTrackbar('kernel size','abs sobel',3,30,nothing)
# kernel_size=11
# cv2.GaussianBlur(warp_grayimg_roi, (kernel_size, kernel_size), 0, warp_grayimg_roi)
# # cv2.medianBlur(warp_grayimg_roi,kernel_size,warp_grayimg_roi)
# binary_absSobelimg = abs_sobel_threshold(warp_grayimg_roi, warp_srcimg_roi, orient='x', thresh_min=100,
#                                          thresh_max=190)
# cv2.imshow('abs sobel', binary_absSobelimg)
# cv2.waitKey(0)
# # while(1):
# #     thresh_min=cv2.getTrackbarPos('min','abs sobel')
# #     thresh_max=cv2.getTrackbarPos('max','abs sobel')
# #     # kernel_size=cv2.getTrackbarPos('kernel size','abs sobel')
# #
# #     # cv2.blur(warp_grayimg_roi,(kernel_size,kernel_size),warp_grayimg_roi)
# #     # cv2.medianBlur(warp_grayimg_roi,kernel_size,warp_grayimg_roi)
# #     cv2.GaussianBlur(warp_grayimg_roi,(kernel_size,kernel_size),0,warp_grayimg_roi)
# #
# #     binary_absSobelimg = abs_sobel_threshold(warp_grayimg_roi, warp_srcimg_roi, orient='x', thresh_min=thresh_min,
# #                                              thresh_max=thresh_max)
# #     cv2.imshow('abs sobel',binary_absSobelimg)
# #     k=cv2.waitKey(1)& 0xFF
# #     if k==27:
# #         break
# # cv2.destroyAllWindows()

#======================================================矩形框
# dst=np.zeros((500,500),dtype=np.uint8)
# dst=np.dstack((dst,dst,dst))
# pts=[50,50,100,100]
# pts2=(150,150,200,200)
# cv2.rectangle(dst,pts[:2],pts[-2:],(255,0,0))
# cv2.imshow('dst',dst)
# cv2.waitKey()

#======================================================三阶多项式拟合直线
# x=np.linspace(0,100-1,100)
# y=x*2+abs(np.random.randn(100))
# y2=x*2
# # print(x)
# # print(y)
# fit=np.polyfit(x,y,3)
# l_fit=fit[0]*x**3+fit[1]*x**2+fit[2]*x+fit[3]
# plt.scatter(x,y)
# plt.plot(x,l_fit,color='yellow')
# plt.show()
# print(fit[0],fit[1],fit[2],fit[3])
