import sys
import cv2
import numpy as np
from pprint import pprint

img = cv2.imread("pana2.jpg",cv2.IMREAD_COLOR)
#cv2.imshow('original', image)

# リサイズ
height, width = img.shape[:2]
#size = (height // 4, width // 4)
#resize = cv2.resize(img, size)
#imageOrg    = image

size = (width, height)
resize = img

# HSVへ変換
hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

# 白抽出：凄く明るい場所
threashhold_min = np.array([0,0,70], np.uint8)
threashhold_max = np.array([255,255,255], np.uint8)
threash = cv2.inRange(hsv, threashhold_min, threashhold_max)

# BGRへ変換
# inRange で グレースケールされている
gray = cv2.cvtColor(threash, cv2.COLOR_GRAY2RGB)


# ノイズ除去
kernel = np.ones((9,9), np.uint8)
mor1   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
mor2   = cv2.morphologyEx(mor1, cv2.MORPH_CLOSE, kernel)

# 反転処理
rev   = 255 - mor2

# 境界抽出
gray_min = np.array([0], np.uint8)
gray_max = np.array([128], np.uint8)
threshold_gray = cv2.inRange(rev, gray_min, gray_max)
contours, hierarchy = cv2.findContours(threshold_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 最大面積を探す
max_area_contour=-1
max_area    = 0
for contour in contours:
    area=cv2.contourArea(contour)
    if(max_area<area):
        max_area=area
        max_area_contour = contour


# カラー化
#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)

contours    = [max_area_contour]

#cv2.drawContours(resize, max_area_contour, -1, (0, 255, 0), 5)



# 輪郭の近似
epsilon = 0.01 * cv2.arcLength(max_area_contour,True)
approx = cv2.approxPolyDP(max_area_contour,epsilon,True)
#print(approx)
#if len(approx) == 4:
#    cv2.drawContours(resize, [approx], -1, (255, 0, 0), 3)

"""
num = 3
for point in approx:
    cv2.circle(resize, (point[0][0],point[0][1]), 100, (255, 0, 0), -1)
    cv2.putText(resize, "1", (point[0][0],point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=num)
    num += 5


cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.imshow("img1", resize)
"""

s_approx = approx[[1,0,2,3],:]

# 射影変換
pts1 = np.float32(s_approx)
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
pprint(pts1)
pprint(pts2)
M = cv2.getPerspectiveTransform(pts1,pts2)
image = cv2.warpPerspective(img,M,size)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", image)
cv2.imwrite("pana_2_1.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
