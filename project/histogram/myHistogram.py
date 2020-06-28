"""
    Dip: Digital image process
    
    def: spatialbalance,直方图均衡算法
    TODO: ,直方图根据目标函数拟合
""" 

import cv2
import numpy
import copy

import sys
sys.path.append("..")
import mytools as tool

# 根据源数组求解直方图分布策略
def spatial(srcarray): 
    total = srcarray.sum()
    targetSpatial = srcarray / total
    #灰度级别树木
    grayNums = len(srcarray) - 1
    _ret = [targetSpatial[0] * grayNums]
    
    for i in range(grayNums + 1):
        _ret.append(0)
        if i == 0: continue
        _ret[i] = (_ret[i-1] + grayNums*targetSpatial[i])
    
    # numpy.narray 下标和 list的下标不一样艾，麻烦！
    _ret = numpy.array(_ret)
    return _ret.astype(int)

# 定义目标图像的直方图分布策略
def gettargetspatial():
    #目标图像像素分布情况
    #targetSpatial = numpy.zeros(255)
    targetSpatial = numpy.arange(1,255)
    #targetSpatial = numpy.array([0,0,0,15,20,30,20,15])
    return  spatial(targetSpatial)



# 获取灰度图像的直方图匹配结果
# 假定处理的图像对象就是8bit的
def densitydistribution(image):
    # 获取灰度图像的密度分布
    # 初始化灰度级数队列，这个队列的下标表示灰度级
    srcGray = numpy.array(numpy.zeros(255), numpy.uint8)
    
    # 一维化image
    Imagedata = image.flatten()
     
     # 统计image的灰度分布
    for i in range(len(Imagedata)):
        srcGray[Imagedata[i] - 1] += 1
        
    # 求原灰度图像的直方图分布策略
    ret1 = spatial(srcGray)
    
    # 获取默认直方图分布策略
    ret2 = gettargetspatial()
    
    # 作直方图拟合
    # 不知道怎么作呢
    for i in range(len(Imagedata)):
        Imagedata[i] = ret1[Imagedata[i]-1]
    
    # 一维转二维
    retImage = Imagedata.reshape(image.shape)    
    
    return retImage

if __name__ == "__main__":
    img = tool.read("sportscar.jpg")
    retImage = densitydistribution(img)
    
    cv2.imshow("src" , img)
    cv2.imshow("ret" , retImage)
    
    cv2.waitKey(0)