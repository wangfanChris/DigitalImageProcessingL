import cv2
import numpy
from matplotlib import pyplot as plt

img = cv2.imread('srcpic/bb.jpg', 0)

img = cv2.resize(img, (600, 600))

fft = numpy.fft.fft2(img)

fshift = numpy.fft.fftshift(fft)

# 对数转浮点数
fftimg = numpy.log(numpy.abs(fft))
shiftimg = numpy.log(numpy.abs(fshift))

rows, cols = img.shape

crow,ccol = int(rows/2), int(cols /2)

mask = numpy.zeros((rows, cols), numpy.uint8)

mask[0:rows, ccol-30:ccol+30] = 1
mask[crow-30:crow+30, 0:cols] = 1

#此处有个问题，掩模后为怎么处理log(0)的？
fshift_mask = fshift*mask 
f_ishift = numpy.fft.ifftshift(fshift_mask)
img_back = numpy.fft.ifft2(f_ishift)
img_back = numpy.abs(img_back)

'''
# 另外一种掩膜
mask = numpy.ones((rows, cols), numpy.uint8)

mask[crow-250:crow+250, ccol-250:ccol+250] = 0
fshift_mask = fshift*mask
fft_mask = fft*mask
img_back = numpy.fft.ifft2(fft_mask)
img_back = numpy.abs(img_back)
'''

# 对数转浮点数
#img_back = numpy.log(numpy.abs(img_back))

plt.subplot(231), plt.imshow(img, 'gray'),
plt.title("imgsrc"), plt.xticks([]), plt.yticks([]) 

plt.subplot(232), plt.imshow(fftimg, 'gray'),
plt.title("fftimg"), plt.xticks([]), plt.yticks([]) 

plt.subplot(233), plt.imshow(shiftimg, 'gray'),
plt.title("shiftimg"), plt.xticks([]), plt.yticks([]) 

plt.subplot(234), plt.imshow(mask, 'gray'),
plt.title("mask"), plt.xticks([]), plt.yticks([]) 

plt.subplot(235), plt.imshow(numpy.log(numpy.abs(fshift_mask)), 'gray'),
plt.title("fshiftfftimgwithMask"), plt.xticks([]), plt.yticks([]) 

plt.subplot(236), plt.imshow(img_back, 'gray'),
plt.title("img_back"), plt.xticks([]), plt.yticks([]) 

plt.show()
