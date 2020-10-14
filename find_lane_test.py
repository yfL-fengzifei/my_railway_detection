"""
find_lane 算法尝试
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from warpImg import *
from binary_threshold import *

def draw_window(img,win_y_low,win_y_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high):
    """
    画窗口
    :param img: 图片
    :param win_y_low: 上边界
    :param win_y_high: 下边界
    :param win_xleft_low: 左窗左边界
    :param win_xleft_high: 左窗右边界
    :param win_xright_low: 右窗左边界
    :param win_xright_high: 右窗左边界
    """
    #结果图像
    if img.shape[-1]!=3:
        dst_img=np.dstack((img,img,img))
    else:
        dst_img=img
    #左右窗
    left_win=(win_xleft_low,win_y_low,win_xleft_high,win_y_high)
    right_win=(win_xright_low,win_y_low,win_xright_high,win_y_high)

    #画矩形
    #**==**还需要再看**==**
    cv2.rectangle(dst_img,left_win[:2],left_win[-2:],(0,0,255),2)
    cv2.rectangle(dst_img,right_win[:2],right_win[-2:],(0,255,0),2)
    # cv2.imshow('dst',dist_img)
    # cv2.waitKey(0)

    return dst_img


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
    # # plt.figure(figsize=(img.shape[0]/100,img.shape[1]/100),dpi=100)
    # plt.plot(histogram)
    # plt.xlim([0,img.shape[1]])
    # # plt.axis('off')
    # plt.savefig('./hist_img2.png',bbox_inches='tight')
    # plt.show()

    #找到左右峰值
    #中点
    midpoint=np.int(histogram.shape[0]//2)
    # print(midpoint)
    # cv2.line(img,(midpoint,img.shape[0]),(midpoint,0),(255,255,0))
    # cv2.imshow('mid line',img)
    # cv2.waitKey(0)
    #四分之一点
    quarter_point=np.int(midpoint//2)
    #np.argmax返回的是索引
    #左峰值点
    # lefx_base=np.argmax(histogram[:midpoint])
    lefx_base=np.argmax(histogram[quarter_point:midpoint])+quarter_point
    # print('left peak point',lefx_base)
    #右峰值点
    # rightx_base=np.argmax(histogram[midpoint:])+midpoint
    rightx_base=np.argmax(histogram[midpoint:(midpoint+quarter_point)])+midpoint
    # print('right peak point',rightx_base)

    #选择滑动窗口的数量
    nwindows=20
    #窗口高度
    window_height=np.int(img.shape[0]/nwindows)
    # print('window_height',window_height)

    #图像中所有非零像素
    nonzero=img.nonzero()
    #非零元素的x,y坐标
    nonzeroy=np.array(nonzero[0]) #第几行
    nonzerox=np.array(nonzero[1]) #第几列

    #当前位置
    leftx_current=lefx_base
    rightx_current=rightx_base
    # print('left current',lefx_current,'right current',rightx_current)

    #设置窗口的正负偏差
    #应该根据实际轨道宽度和像素宽度进行设置
    # print((rightx_base-lefx_base)/2)
    margin=60
    #Set minimum number of pixels found to recenter window
    minpix=40

    #创建空列表来接收左右车道的索引
    left_lane_inds=[]
    right_lane_inds=[]

    #遍历窗口
    for window in range(nwindows):
        #定义窗口的边界
        #上边界
        win_y_low=img.shape[0]-(window+1)*window_height
        #下边界
        win_y_high=img.shape[0]-(window)*window_height

        #左窗
        #左边界
        win_xleft_low=leftx_current-margin
        win_xleft_high=leftx_current+margin
        #右窗
        #右边界
        win_xright_low=rightx_current-margin
        win_xright_high=rightx_current+margin

        #为了显示窗口(提前测试)
        if window==0:
            dst_img=img.copy() #深拷贝
        dst_img=draw_window(dst_img,win_y_low,win_y_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high)

        #寻找窗口中满足要求(非零)的索引
        #len(nonzeroy)=len(nonzerox)
        #左轨道
        good_left_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        #右轨道
        good_right_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
        #添加到列表中
        #每个元素作为一个窗口的索引值(与x和y无关)
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #如果满足要求的像素的数量很多，就重新更新当前窗口坐标
        if len(good_left_inds)>minpix:
            leftx_current=np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
            rightx_current=np.int(np.mean(nonzerox[good_right_inds]))

    #为了显示滑动窗口
    # cv2.imshow('dst window',dst_img)
    # cv2.waitKey(0)

    #连接索引
    #默认axis=0
    left_lane_inds=np.concatenate(left_lane_inds)
    right_lane_inds=np.concatenate(right_lane_inds)

    #提取坐标
    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
    # #散点测试
    # plt.scatter(leftx,lefty,marker='+')
    # plt.scatter(rightx,righty,marker='^')
    # plt.xlim([0,500])
    # ax = plt.gca()
    # ax.xaxis.set_ticks_position('top')
    # ax.invert_yaxis()
    # plt.show()

    return leftx,lefty,rightx,righty,dst_img


def fit_lane(leftx,lefty,rightx,righty,binary_img,dst_win_img):
    """
    拟合轨道
    :param leftx: 左道x坐标
    :param lefty: 左道y坐标
    :param rightx: 右道x坐标
    :param righty: 右道y坐标
    :param binary_img: 二值图像 单通道
    :param dst_win_img: 带滑动窗口图像 三通道
    :return: 拟合的轨道，左和右
    """
    #多项式拟合返回的是多项式系数
    # #二阶多项式拟合
    # left_fit=np.polyfit(lefty,leftx,2)
    # right_fit=np.polyfit(righty,rightx,2)
    # #多项式
    # ploty=np.linspace(0,binary_img.shape[0]-1,binary_img.shape[0])
    # left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    # right_fitx=right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    #三阶多项式拟合
    left_fit=np.polyfit(lefty,leftx,3)
    right_fit=np.polyfit(righty,rightx,3)
    #多项式
    ploty=np.linspace(0,binary_img.shape[0]-1,binary_img.shape[0])
    left_fitx=left_fit[0]*ploty**3+left_fit[1]*ploty**2+left_fit[2]*ploty+left_fit[3]
    right_fitx=right_fit[0]*ploty**3+right_fit[1]*ploty**2+right_fit[2]*ploty+right_fit[3]

    #显示
    #注意索引顺序
    # dst_win_img[lefty,leftx]=[255,0,0]
    # dst_win_img[righty,rightx]=[0,0,255]
    # plt.imshow(dst_win_img)
    # plt.plot(left_fitx,ploty,color='yellow',linewidth=4.0)
    # plt.plot(right_fitx,ploty,color='red',linewidth=4.0)
    # plt.show()

    return left_fitx,right_fitx

def fit_dirvearea(src_img,warp_grayimg_roi,binary_img,Min,left_fitx,right_fitx):
    """
    画行驶区域
    :param src_img: 原始图像
    :param warp_grayimg_roi: 原始图像+ roi
    :param binary_img: roi
    :param Min: 逆透视变换矩阵
    :param left_fitx: 左道拟合
    :param right_fitx: 右道拟合
    :return:
    """
    warp_zero=np.zeros_like(warp_grayimg_roi).astype(np.uint8)
    color_warp=np.dstack((warp_zero,warp_zero,warp_zero))
    warp_roi_zero=np.zeros_like(binary_img).astype(np.uint8)
    color_roi_warp=np.dstack((warp_roi_zero,warp_roi_zero,warp_roi_zero))

    ploty=np.linspace(0,binary_img.shape[0]-1,binary_img.shape[0])
    pts_left=np.array([np.transpose(np.vstack([left_fitx,ploty]))]) #这里转换维度变成一一对应(n,2) 2个维度
    pts_right=np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts=np.hstack((pts_left,pts_right)) #为什么...???...

    #利用多边形画出图像
    cv2.fillPoly(color_roi_warp,np.int_([pts]),(0,255,0))
    # cv2.imshow('area',color_roi_zero)
    # cv2.waitKey(0)
    color_warp[:,450:950,:]=color_roi_warp
    # cv2.imshow('area in src_shape',color_warp)
    # cv2.waitKey(0)

    #显示在原图上
    area_map=cv2.warpPerspective(color_warp,Min,(src_img.shape[1],src_img.shape[0]))
    # cv2.imshow('area map',area_map)
    # cv2.waitKey(0)
    dst_img=src_img.copy()
    dst_img=cv2.addWeighted(dst_img,1,area_map,0.3,0)
    cv2.imshow('dst',dst_img)
    cv2.waitKey(0)
    pass


#main
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
#将ROI保存为图像
warp_grayimg_roi_copy=warp_grayimg_roi[:,450:950]
# cv2.imshow('cpoy',warp_grayimg_roi_copy)
# cv2.waitKey(0)
warp_srcimg_roi_copy=warp_srcimg_roi[:,450:950,:]
# cv2.imshow('copy',warp_srcimg_roi_copy)
# cv2.waitKey(0)

#阈值分割
# binary_img = binary_threshold(warp_grayimg_roi, warp_srcimg_roi)
#仅对ROI做分割
binary_img = binary_threshold(warp_grayimg_roi_copy, warp_srcimg_roi_copy)

#查找索引
leftx,lefty,rightx,righty,dst_win_img=find_lane(binary_img)
# cv2.imshow('dst',dst_win_img)
# cv2.waitKey(0)
#直方图分析
# hist_img=cv2.imread('./hist_img.png')
# cv2.imshow('hist',hist_img)
# print(hist_img.shape)
# resize_factor=500/hist_img.shape[1]
# hist_img=cv2.resize(hist_img,(0,0),fx=resize_factor,fy=resize_factor)
# print(hist_img.shape)
# binary_img_convert=np.dstack((binary_img,binary_img,binary_img))
# print(binary_img_convert.shape)
# # cv2.imshow('binary img convert',binary_img_convert)
# # cv2.waitKey(0)
# # addweight_hist_img=cv2.addWeighted(binary_img_convert[-350:,:,:],0.3,hist_img,0.7,0)
# add_hist=np.zeros_like(binary_img_convert)
# add_hist[-hist_img.shape[0]:,:,:]=hist_img
# addweight_hist_img=cv2.addWeighted(binary_img_convert,0.3,add_hist,0.7,0)
# cv2.imshow('add result',addweight_hist_img)
# cv2.waitKey(0)

#曲线拟合并显示
left_fitx,right_fitx=fit_lane(leftx,lefty,rightx,righty,binary_img,dst_win_img)

#绘制行驶区域
fit_dirvearea(src_img,warp_grayimg_roi,binary_img,Min,left_fitx,right_fitx)



