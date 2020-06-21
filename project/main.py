import sys
import os
import cv2
import numpy
import matplotlib

def showVersion():
    print('python version ' + sys.version)
    print('opencv version ' + cv2.__version__)
    print('numpy version ' + numpy.__version__)
    print('matplotlib version ' + matplotlib.__version__)

if __name__ == "__main__":
    showVersion()