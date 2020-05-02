import sys
import cv2
import numpy as np

image = cv2.imread('pana2.jpg')

cv2.namedWindow("original")
cv2.moveWindow("original", 200, 200)
cv2.imshow('original', image)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)
#kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)
print(kernel)

dst = cv2.filter2D(image, -1, kernel)

cv2.namedWindow("result")
cv2.moveWindow("result", 200, 400)
cv2.imshow('result', dst)
cv2.imwrite('pana2_0.jpg', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
