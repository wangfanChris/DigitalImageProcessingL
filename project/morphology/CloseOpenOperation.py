#define open & close method

import cv2
def open_image(img):
    # define kernel 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return binary

# para img need pretrans to binary image
def close_image(img):
    # define kernel 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
    return binary

if __name__ == "__main__":
            
    img = cv2.imread('../srcpic/bb.jpg')

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("src", img)
    # Trans to 2 value image
    ret, binary = cv2.threshold(img, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)

    open_binary = open_image(binary)
    cv2.imshow('open_binary', open_binary)

    close_binary = close_image(binary)
    cv2.imshow('close_binary', close_binary)

    open_close = close_image(open_binary)
    cv2.imshow('open_close', open_close)

    close_open= open_image(close_binary)
    cv2.imshow('close_open', close_open)

    cv2.waitKey(0)




 