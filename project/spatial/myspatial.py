
import numpy
import cv2
import copy
import sys

sys.path.append('..')
import mytools
import filter
 
def main():
    filterInstance = filter.SimpleMedianFilter() 
    img = mytools.read("moon.tif")
    cv2filter = filterInstance.apply(img)  
    myapply = filterInstance.myapply(img)
    diff = numpy.abs(myapply - cv2filter)
    mytools.drawlocalImage([img, cv2filter, myapply, diff], locals())
    pass    

if __name__ == "__main__":
    main()