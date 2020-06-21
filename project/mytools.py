"""
    Mytools introduction:
    functional 1. image import & export
    TODO functional 2. draw a list filled by images
"""
import cv2
from matplotlib import pyplot as plt
import os

input_abs_directory = "/home/wangfan/Desktop/dip/srcpic/"
input_directory = "srcpic/"

output_abs_directory= "/home/wangfan/Desktop/dip/outputpic/"
output_directory = "outputpic/"

#==========================================================
#Image import / export
#==========================================================

def read_abs(fileName):
    img = cv2.imread(input_abs_directory + fileName, 0)
    if img is None:
        print("Alert! cv2.imread Image Error. Plz try to adjust file path")
        return None
    return img

def read(fileName):
    path = os.path.dirname(__file__) + "/"
    img = cv2.imread(path + input_directory + fileName, 0)
    if img is None:
        print("Alert! cv2.imread Image Error. Plz try to adjust file path")
        return None
    return img

def save_abs(image, fileName, fileType = None): 
    return cv2.imwrite(output_abs_directory + fileName + fileType, image)

def save(image, fileName, fileType = None):
    return cv2.imwrite(output_directory + fileName + fileType, image)


#==========================================================
#Image printer
#==========================================================





#test 
if __name__ == "__main__":
    cv2.imshow("T1",  read("cars.jpg"))
    cv2.imshow("T2",  read_abs("car.jpg")) 
    cv2.waitKey(0)