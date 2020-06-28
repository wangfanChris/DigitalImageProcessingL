import cv2
import numpy
from skimage import exposure

import sys
sys.path.append("..")
import mytools as tool

def histogramEqualize(image, maxIntensity):
    rows, cols = image.shape
    
    # get cdf from image
    cdf, binCenters = exposure.cumulative_distribution(image, maxIntensity)
    binCenters = binCenters.tolist()
    
    for i in range(rows) :
        for j in range(cols):
            
            try:
                probability = cdf[binCenters.index(image[i][j])]
            except:
                probability = 1
            
            image[i][j] = int(probability * maxIntensity)
            
    return image, binCenters, cdf


if __name__ == "__main__":
    image = tool.read("sportscar.jpg")
    cv2.imshow("image", image)
    maxIntensity = 255 # 8 bits
    trans, binCenters, cdf = histogramEqualize(image, maxIntensity)
     
    cv2.imshow("trans", trans)
    
    cv2.waitKey(0)