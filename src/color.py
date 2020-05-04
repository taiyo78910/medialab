import cv2
import numpy as np

# 0 <= h <= 179 (色相)　R:0(180),G:60,B:120
# 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
# 0 <= v <= 255 (明度)　大きいと明るく，小さいと暗い
Rl_LOW_COLOR = np.array([0, 35, 0])
Rl_HIGH_COLOR = np.array([30, 255, 255])
Rh_LOW_COLOR = np.array([150, 20, 0])
Rh_HIGH_COLOR = np.array([179, 255, 255])
G_LOW_COLOR = np.array([30, 40, 50])
G_HIGH_COLOR = np.array([90, 255, 255])
B_LOW_COLOR = np.array([90, 20, 45])
B_HIGH_COLOR = np.array([150, 255, 255])

# 抽出する色の塊のしきい値
#AREA_RATIO_THRESHOLD = 0.005

frame = cv2.imread("pana_2_1.jpg")

h,w,c = frame.shape

# hsv色空間に変換
hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.imshow('img1',hsv)

# 色を抽出する
rl_ex_img = cv2.inRange(hsv,Rl_LOW_COLOR,Rl_HIGH_COLOR)
rh_ex_img = cv2.inRange(hsv,Rh_LOW_COLOR,Rh_HIGH_COLOR)
g_ex_img = cv2.inRange(hsv,G_LOW_COLOR,G_HIGH_COLOR)
b_ex_img = cv2.inRange(hsv,B_LOW_COLOR,B_HIGH_COLOR)

# 輪郭抽出
rl_contours,rl_hierarchy = cv2.findContours(rl_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rh_contours,rh_hierarchy = cv2.findContours(rh_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
g_contours,g_hierarchy = cv2.findContours(g_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
b_contours,b_hierarchy = cv2.findContours(b_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print(g_contours)

min_area = frame.shape[0] * frame.shape[1] * 1e-4
tmp = frame.copy()
h,w,c = frame.shape

contours_list = [rl_contours,rh_contours,g_contours,b_contours]
colors = [(0,0,255),(0,0,255),(0,255,0),(255,0,0)]

for  contours, rect_color in zip(contours_list, colors):
    if len(contours) > 0:
        for i, contour in enumerate(contours):
            rect = cv2.boundingRect(contour)
            if rect[2] < 10 or rect[3] < 2:
                continue
            if rect[3] > 30:
                continue
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            cv2.rectangle(tmp, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), rect_color, 2)

cv2.imwrite("pana_2_2.jpg", tmp)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow('img',tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()
