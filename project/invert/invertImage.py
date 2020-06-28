"""
    Dip: Digital image process
    
    invert image model

    function1: resize()

""" 

import cv2
import numpy 
import copy
 
#==========================================================
#Image resize()     
# 双线性内插法
# 近临内插法
# #==========================================================

EU_NearNeiborInter = 0
EU_DblLineInter = 1

# 默认是255 bit8位的image
def  resize_nei(image, shape) :
    # 不希望外界的bug定位到这里来
    if image is None: return None
   # if not isinstance(image.dtype, numpy.uint8) : return None
    colors, bits = 258, 8
    rows, cols = image.shape
    newRows, newCols = shape

    # 计算纵横放缩比
    enlargerate_rows, enlargerate_cols = (rows / newRows), (cols / newCols)

    # 初始化代表新图像的数组
    tarArr = numpy.zeros((newRows, newCols), dtype=numpy.uint8)

    for row in range(newRows - 1):
        for col in range(newCols - 1):
            index_row = round(row * enlargerate_rows)
            index_col = round(col * enlargerate_cols)
            if index_row == 0: index_row =1
            if index_row >= rows: index_row = rows - 1
            if index_col == 0: index_col == 1
            if index_col >= cols: index_col = cols -1
            tarArr[row][col] = image[index_row][index_col]
    return tarArr

def resize_ner(image, shape):
    # 不希望外界的bug定位到这里来
    if image is None: return None
   # if not isinstance(image.dtype, numpy.uint8) : return None
    colors, bits = 258, 8
    rows, cols = image.shape
    newRows, newCols = shape

    # 计算纵横放缩比
    enlargerate_rows, enlargerate_cols = (rows / newRows), (cols / newCols)

    # 初始化代表新图像的数组
    tarArr = numpy.zeros((newRows, newCols), dtype=numpy.uint8)

    kernel = numpy.array([[0,0],[0, 0],[0, 0],[0, 0]])
    index_row, index_col = 0,0
    u,v = 0.1,0.1

    for row in range(newRows - 1):
        for col in range(newCols - 1):
            index_row = round(row * enlargerate_rows)
            u = row * enlargerate_rows - index_row
            index_col = round(col * enlargerate_cols)
            v = col * enlargerate_cols - index_col
            if index_row == 0: index_row =0
            if index_row >= rows - 1: index_row = rows - 2
            if index_col == 0: index_col == 0
            if index_col >= cols - 1: index_col = cols -2

            # 以相邻的四个各自作为依据
            kernel[:]  = (index_row, index_col), (index_row, index_col + 1), (index_row +1, index_col), (index_row+1, index_col+1)
    
            tarArr[row][col] = (1 - u)*(1-v) * image[kernel[0, 0]][kernel[0, 1]] + \
                (1 - u)*v * image[kernel[0, 0]][kernel[0, 1]] + \
                    u*(1-v) * image[kernel[0, 0]][kernel[0, 1]] + \
                        u*v* image[kernel[0, 0]][kernel[0, 1]]

    return tarArr

# 入口
def resize(image, shape , method = EU_DblLineInter): 
    if method == EU_DblLineInter: return resize_nei(image)
    if method == EU_DblLineInter: return resize_ner(image)

if __name__ == "__main__":
    img = cv2.imread("/home/wangfan/Desktop/dip/srcpic/car.jpg", 0)
    a1 = resize_nei(img, (200, 250))
    a1 = resize_nei(a1, (100, 150))
    a1 = resize_nei(a1, img.shape)
    a2 = resize_ner(img, (200, 250))
    a2 = resize_ner(a2, (100, 150))
    a2 = resize_ner(a2, img.shape)

    cv2.imshow("img", img)
    if a1 is not None :cv2.imshow("a1", a1)
    if a2 is not None :cv2.imshow("a2", a2)

    cv2.waitKey(0)