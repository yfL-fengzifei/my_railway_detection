import cv2
import numpy as np

def warpImg(img):
    """
    透视变换与逆透视变换
    后期可修改坐标
    :param img: ndarry img
    :param src_pts: 原始图像的点
    :param dst_pts: 逆透视变换后的点
    :return: 逆透视变换后的图像，鸟瞰图
    """
    #原始图像维度
    img_size=img.shape[:2] #(h,w)

    #透视\逆透视变换矩阵
    src_pts=np.float32([[(400, 720), (566, 200), (666, 200), (1066, 720)]])
    dst_pts=np.float32([[(450,720),(450,0),(950,0),(950,720)]])

    M=cv2.getPerspectiveTransform(src_pts,dst_pts)
    Min=cv2.getPerspectiveTransform(dst_pts,src_pts)

    return M,Min