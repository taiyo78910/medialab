import cv2
import numpy as np
import sys

input_original_data = '2.jpg'
img = cv2.imread(input_original_data)
h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

size = (5, 5)
blur = cv2.GaussianBlur(gray, size, 0)


lap = cv2.Laplacian(blur, cv2.CV_8UC1)

ret2, th2 = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((5, 20), np.uint8)
closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)


kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(closing, kernel, iterations = 1)
erosion = cv2.erode(dilation, kernel, iterations = 1)


cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow('img',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
