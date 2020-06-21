"""
    Dip: Digital image process
    
    def: dft,
    TODO: 只通过相位或频谱恢复图像实验
    TODO: 通过

""" 
import sys
sys.path.append("..")

import cv2
import numpy
import math
import mytools as tool
import copy
from matplotlib import pyplot as plt

def singleRowDft(row):
    X = [numpy.complex(0)] * len(row)

    n = float(len(row))

    for u in range(len(row)):
        curSum = numpy.complex(0)

        for v in range(len(row)):
            curOperand = row[v] * numpy.exp((-2j * math.pi * u * v) / n)
            curSum = curSum + curOperand
        
        X[u] = curSum
    
    return X


def myDFT(image):
    '''
    fourier tranform of an image
    '''
    width, height = image.shape

    horizonDFT = numpy.asarray([[numpy.complex(0)] * width] * height)
    verticalAndHorizonDft = copy.deepcopy(horizonDFT)

    # rows
    counters = 0
    for row in image:
        print("row" + str(counters + 1) + "of" + str(image.shape[0]) + "...")
        horizonDFT[counters] = singleRowDft(row)
        counters += 1
    
    counters = 0
    for col in horizonDFT.T :
        print("\t" + "row" + str(counters + 1) + "of" + str(image.shape[0]) + "...")
        verticalAndHorizonDft.T[counters] = singleRowDft(col)
        counters += 1

    return verticalAndHorizonDft

if __name__ == "__main__":
    srcImg = tool.read("car.jpg")

    srcImg = cv2.resize(srcImg, (100, 100))

    # my频率化图像    
    mydft = myDFT(srcImg)
    #格式化显示
    mydft_Draw = numpy.log(numpy.abs(numpy.fft.fftshift(mydft)))

    #numpy 频率化图像
    npDft = numpy.fft.fft2(srcImg)
    npdft_Draw = numpy.log(numpy.abs(numpy.fft.fftshift(npDft)))
 
    diff = (npdft_Draw - mydft_Draw)

    plt.subplot(231), plt.imshow(srcImg, 'gray'),
    plt.title("srcImg"), plt.xticks([]), plt.yticks([]) 

    plt.subplot(232), plt.imshow(mydft_Draw, 'gray'),
    plt.title("mydft_Draw"), plt.xticks([]), plt.yticks([]) 

    plt.subplot(233), plt.imshow(npdft_Draw, 'gray'),
    plt.title("npdft_Draw"), plt.xticks([]), plt.yticks([]) 

    plt.subplot(234), plt.imshow(diff, 'gray'),
    plt.title("diff"), plt.xticks([]), plt.yticks([]) 
    plt.show()
    cv2.waitKey(0)