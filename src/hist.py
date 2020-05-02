import numpy as np
import cv2

img = cv2.imread('t_full_1.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

img1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.imshow('img1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()