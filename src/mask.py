import cv2
import numpy as np

# 赤色の検出
def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,64,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,64,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）
    mask = mask1 + mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 緑色の検出
def detect_green_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 緑色のHSVの値域1
    hsv_min = np.array([20, 35, 0])
    hsv_max = np.array([80,255,255])

    # 緑色領域のマスク（255：赤色、0：赤色以外）
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 青色の検出
def detect_blue_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSVの値域1
    hsv_min = np.array([90, 64, 0])
    hsv_max = np.array([150,255,255])

    # 青色領域のマスク（255：赤色、0：赤色以外）
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img


# 入力画像の読み込み
img = cv2.imread("21.jpg")

# 色検出（赤、緑、青）
red_mask, red_masked_img = detect_red_color(img)
green_mask, green_masked_img = detect_green_color(img)
blue_mask, blue_masked_img = detect_blue_color(img)

# 結果を出力
cv2.imwrite("red_mask.jpg", red_mask)
cv2.imwrite("red_masked_img.jpg", red_masked_img)
cv2.imwrite("green_mask.jpg", green_mask)
cv2.imwrite("green_masked_img.jpg", green_masked_img)
cv2.imwrite("blue_mask.jpg", blue_mask)
cv2.imwrite("blue_masked_img.jpg", blue_masked_img)

dst1 = cv2.addWeighted(red_masked_img, 0.5, green_masked_img, 0.5, 0)
dst = cv2.addWeighted(blue_masked_img, 0.5, dst1, 0.5, 0)

cv2.imwrite('add_weighted.jpg', dst)
