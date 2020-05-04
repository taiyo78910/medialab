import cv2  # OpenCVのインポート

img = cv2.imread('pana2.jpg')  # 画像の読み出し
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  # 色空間をBGRからHSVに変換
s_magnification = 1.5  # 彩度(Saturation)の倍率
v_magnification = 0.7  # 明度(Value)の倍率

img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*s_magnification  # 彩度の計算
img_hsv[:,:,(2)] = img_hsv[:,:,(2)]*v_magnification  # 明度の計算
img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)  # 色空間をHSVからBGRに変換

#cv2.imwrite('differ1.jpg', img_bgr)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow('img',img_bgr)
cv2.imwrite('pana2_0.jpg', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
