import numpy
import cv2 
import copy
import math

import sys
sys.path.append("..")
import mytools

def normalize(src):   
    _range = numpy.max(src) - numpy.min(src)
    _min = numpy.min(src)
    return   (abs((src - _min) / _range * 254)).astype(numpy.uint8)


class FrequenceFilter(object):
    def __init__(self):pass
    
    def apply(self, src):
        # 求傅里叶变换谱
        fftl = numpy.fft.fft2(src)
        # 高频中心化
        sfftl = numpy.fft.fftshift(fftl)
        
        dstsfftl = self.frequencyAlgorithm(sfftl)
        
        dstfftl = numpy.fft.fftshift(dstsfftl)
        dst = numpy.fft.ifft2(dstfftl)
        
        # 归一化
        sfftl = numpy.abs(numpy.log(sfftl))
        dstsfftl = numpy.abs(numpy.log(dstsfftl))
        dst = numpy.abs(numpy.log(dst))
        
        return sfftl, dstsfftl, dst
    
# 带阻滤波   高频和低频通过
class BandStopFilter(FrequenceFilter):
    def __init__(self) : FrequenceFilter.__init__(self)
    
    def frequencyAlgorithm(self, sfftl):
        rows,cols = sfftl.shape   
        dstfftl = copy.deepcopy(sfftl)
         
        d0 = 50
        d1 = 250
        for i in range(rows):
            for j in range(cols):
                dis = math.sqrt((i - rows / 2)**2 + (j - cols / 2)**2)
                if (dis <= d0 or dis >= d1): h = 1
                else: 
                    h = 0
                dstfftl[i,j] = h * sfftl[i, j]
                
        return dstfftl


# 带阻滤波   高频和低频通过
class BandPassFilter(FrequenceFilter):
    def __init__(self) : FrequenceFilter.__init__(self)
    
    def frequencyAlgorithm(self, sfftl):
        rows,cols = sfftl.shape   
        dstfftl = copy.deepcopy(sfftl)
         
        d0 = 0
        d1 = 200
        
        for i in range(rows):
            for j in range(cols):
                dis = math.sqrt((i - rows / 2)**2 + (j - cols / 2)**2)
                if (dis <= d0 or dis >= d1): h = 0
                else: 
                    h = 1
                dstfftl[i,j] = h * sfftl[i, j]
        
        maskwidth = 10
        crow,ccol = int(rows/2), int(cols /2)
        mask = numpy.ones((rows, cols), numpy.uint8)
        
        mask[0:rows, ccol-maskwidth:ccol+maskwidth] = 0
        mask[crow-maskwidth:crow+maskwidth, 0:cols] = 0
        cv2.imshow("sddd", mask * 244)
        return dstfftl * mask
    
# 带通滤波   高频和低频不通过 中频通过
        
if __name__ == "__main__":
    filterInstance = BandPassFilter()
    img = mytools.read("moon.tif")
    img = cv2.resize(img, (400, 400))
    
    # 三个图像分别是 归一化的 源图像傅里叶变换， 叠加滤波器的傅里叶变化， 和目标图像
    sfftl, dstsfftl, dst = filterInstance.apply(img)  
    
    mytools.drawlocalImage([img, sfftl, dstsfftl, dst, img - dst], locals())
    pass    
