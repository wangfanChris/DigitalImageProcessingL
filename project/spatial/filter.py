import numpy
import cv2 
import copy

# 滤波器归一化函数
def kernelNormalized(kernel):
    return kernel / numpy.sum(kernel)
# 卷积基类
class ConvolutionFilter(object):    
    def __init__(self, kernel):
        self._kernel = kernel
    
    def apply(self, src):
        dst = copy.deepcopy(src)
        cv2.filter2D(src, -1, self._kernel, dst)
        return dst
    
    def myapply(self, src):
            width, heigh = src.shape
            N = self._kernel.shape[0]
            addW = int (N  / 2)
        
            #asarray 不会拷贝新的副本， array会拷贝新的副本，效率低下
            copyImg = numpy.zeros((width + 2 * addW, heigh+ 2 * addW), \
                numpy.uint8) 
            destImg = copy.deepcopy(copyImg)
            copyImg[addW: addW + width, addW: addW+ heigh] = src
              
            for i in range(width):
                for j in range(heigh):
                    copyKernel = copyImg[i: i + N , j:j + N]
                    destImg[i][j] = int(numpy.sum(copyKernel * self. _kernel))
            return destImg[0: width, 0:heigh]

#简单均值滤波5x5
class SimpleAverage5x5(ConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]])
        ConvolutionFilter.__init__(self, kernelNormalized(kernel))
        
#简单均值滤波3x3
class SimpleAverage3x3(ConvolutionFilter):
    def __init__(self):                
        kernel = numpy.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        ConvolutionFilter.__init__(self, kernelNormalized(kernel))

#中心权重均值滤波
class CenterHighAverage(ConvolutionFilter):
    def __init__(self):
        kernel = numpy.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]
        )
        ConvolutionFilter.__init__(self, kernelNormalized(kernel))

# 拉普拉斯算子
class LaplceOperator:
    def __init__(self):
        kernel1 = numpy.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]
        )
        kernel2 = numpy.array(
            [
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ]
        )
        kernel3 = numpy.array(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ]
        )
        kernel4 = numpy.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        )  
        self._kernel1 = kernel1
        self._kernel2 = kernel1
        self._kernel3 = kernel1
        self._kernel4 = kernel1
    
    def privateapply(self, src, kernel):
        con = ConvolutionFilter(kernel)
        return con.myapply(src)
    
    def apply(self, src):
        return src + cv2.Laplacian(src, cv2.CV_16S, ksize = 3)
    
    def myapply(self, src):
        img1 = self.privateapply(src, self._kernel1)
        img2 = self.privateapply(src, self._kernel2)
        img3 = self.privateapply(src, self._kernel3)
        img4 = self.privateapply(src, self._kernel4)
        
        cv2.imshow("img1", img1 - src)
        cv2.imshow("img2", img2 - src)
        cv2.imshow("img3", img3 - src)
        cv2.imshow("img4", img4 - src)
        return img1
    
# 非锐化掩蔽
class UnSharpFilter():
    def __init__(self):
        self._Unsharpfilter = SimpleAverage3x3()
    def apply(self, src): 
        #src = self._Unsharpfilter.apply(src)
        unsharp = self._Unsharpfilter.apply(src)
        diff = src - unsharp
        dst = src + diff * 2
        ##cv2.imshow("src", src)
        #cv2.imshow("unsharp", unsharp)
        #cv2.imshow("diff", diff)
        #cv2.imshow("dst", dst)
        return dst
    
    def myapply(self, src):
        unsharp = self._Unsharpfilter.myapply(src)
        diff = src - unsharp
        dst = src + diff
        return dst

# 图像梯度锐化（非线性）
class GradientSharping:
    # 此滤波器的中心点输出是两卷积之和，因此要取两次
    def __init__(self):
        kernelpart1 = numpy.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]            
        ) 
        kernelpart2 = numpy.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]            
        ) 
        
        self._conpart1 = ConvolutionFilter(kernelpart1)
        self._conpart2 = ConvolutionFilter(kernelpart2)
        
    def apply(self, src):
        return numpy.abs(self._conpart1.apply(src)) +\
             numpy.abs(self._conpart2.apply(src))
    def myapply(self, src):
        return numpy.abs(self._conpart1.myapply(src)) +\
             numpy.abs(self._conpart2.myapply(src))

# 几何均值滤波
# 这里的计算有越界倾向，不知道如何处理。
class GeometricMean:
    def __init__(self): 
        self._kernel = numpy.array([[1,1,1], [1, 1, 1], [1, 1, 1]])
    def myapply(self, src):
        rows, cols = src.shape
        dst = numpy.zeros((rows, cols), numpy.float)
        for i in range(rows - 1):
            for j in range(cols - 1):
                cal =  copy.deepcopy(src[i : i + 2, j : j + 2])
                prod =numpy.prod(cal)
                dst[i+1][j+1] =  pow(prod, 1/4)
        return dst
    
    def apply(self, src): 
        return src

 # ========================================================
 # 统计滤波器
 # ========================================================
class StatisticFilterBase(object):
    def __init__(self): pass 
        
    def apply(self, src, kersize = 3):
        dst = numpy.zeros(src.shape, numpy.uint8)
        for i in range(src.shape[0] - kersize + 1):
            for j in range(src.shape[1] - kersize + 1):
                cal = src[i : i + kersize, j : j + kersize]
                dst[i][j] = self.statisticAlgorithm(cal)
        return dst
    
    def myapply(self, src, kersize = 3): return src
        
class SimpleMedianFilter(StatisticFilterBase):
     def __init__(self):
         StatisticFilterBase.__init__(self)
         
     def statisticAlgorithm(self, cal):
         return numpy.median(cal)

class SimpleMaxFilter(StatisticFilterBase):
    def __init__(self):
        StatisticFilterBase.__init__(self)
        
    def statisticAlgorithm(self, cal):
        return numpy.max(cal)
    
class SimpleMinFilter(StatisticFilterBase):
    def __init__(self):
        StatisticFilterBase.__init__(self)
        
    def statisticAlgorithm(self, cal):
        return numpy.min(cal)
    
class SimpleMidFilter(StatisticFilterBase):
    def __init__(self):
        StatisticFilterBase.__init__(self)
        
    def statisticAlgorithm(self, cal):
        return int(numpy.min(cal) / 2 + numpy.max(cal) / 2)

