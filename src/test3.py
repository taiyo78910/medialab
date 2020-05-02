import cv2
import numpy as np

frame = cv2.imread("t_img.jpg")

cv2.imwrite('img.jpg', frame)
