# Erosion 腐蚀 Dilation膨胀
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2. imread('../srcpic/cars.jpg', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# erosion
img_eroded = cv2.erode(img, kernel)
# dilation
img_dilated = cv2.dilate(img, img_eroded)
  
NpKernel = np.uint8(np.ones((3,3)))
Nperoded = cv2.erode(img, NpKernel)
 
plt.subplot(231), plt.imshow(img, 'gray'),
plt.title("imgsrc"), plt.xticks([]), plt.yticks([]) 

plt.subplot(232), plt.imshow(img_eroded, 'gray'),
plt.title("img_eroded"), plt.xticks([]), plt.yticks([]) 

plt.subplot(233), plt.imshow(img_dilated, 'gray'),
plt.title("img_dilated"), plt.xticks([]), plt.yticks([]) 

plt.subplot(234), plt.imshow(Nperoded, 'gray'),
plt.title("Nperoded"), plt.xticks([]), plt.yticks([]) 
 
plt.show()






