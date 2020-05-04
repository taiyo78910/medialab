import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def adjust_edge(img):
    # リサイズ
    height, width = img.shape[:2]

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

    # 輪郭の近似
    epsilon = 0.01 * cv2.arcLength(max_area_contour,True)
    approx = cv2.approxPolyDP(max_area_contour,epsilon,True)

    s_approx = approx[[1,0,2,3],:]

    # 射影変換
    pts1 = np.float32(s_approx)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,size)

    return(img)

def cvt_saturation(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    roi = (100,100,1350,1000)
    s_roi = img_hsv[roi[1]: roi[3], roi[0]: roi[2]]
    s_rate = 150 / np.mean(s_roi[:,:,(2)])
    img_hsv[:,:,(2)] = img_hsv[:,:,(2)] * s_rate
    img = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    return(img)

def cvt_brightness(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    roi = (100,100,1350,1000)
    s_roi = img_hsv[roi[1]: roi[3], roi[0]: roi[2]]
    b_rate = 150 / np.mean(s_roi[:,:,(2)])
    img_hsv[:,:,(2)] = img_hsv[:,:,(2)] * b_rate
    img = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    return(img)

def extract_bgr(img):
    height, width, channels = img.shape[:3]       
    zeros = np.zeros((height, width), img.dtype)  
    b_img_c1, g_img_c1, r_img_c1 = cv2.split(img)
    b_img = cv2.merge((b_img_c1, zeros,zeros))
    g_img = cv2.merge((zeros, g_img_c1, zeros))
    r_img = cv2.merge((zeros, zeros, r_img_c1))
    
    return (b_img, g_img, r_img)

def 

def plot_hist(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def plot_ratio(img):
    r = 1e-10
    hist_list = []
    color = ('b','g','r')
    for i,col in enumerate(color):
        hist_list.append(cv2.calcHist([img],[i],None,[256],[0,256]))
    
    for j,col in enumerate(color):
        ratio = (hist_list[j] / (hist_list[0] + hist_list[1] + hist_list[2] + r))
        plt.plot(ratio,color = col)
        plt.xlim([0,256])
    plt.show()



def clahe(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return(img)

def get_position(img):
    # 0 <= h <= 179 (色相)　R:0(180),G:60,B:120
    # 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
    # 0 <= v <= 255 (明度)　大きいと明るく，小さいと暗い
    Rl_LOW_COLOR = np.array([0, 50, 130])
    Rl_HIGH_COLOR = np.array([30, 255, 255])
    Rh_LOW_COLOR = np.array([150, 0, 0])
    Rh_HIGH_COLOR = np.array([179, 255, 255])
    G_LOW_COLOR = np.array([30, 55, 30])
    G_HIGH_COLOR = np.array([90, 255, 255])
    B_LOW_COLOR = np.array([90, 0, 0])
    B_HIGH_COLOR = np.array([150, 255, 255])

    # 抽出する色の塊のしきい値
    #AREA_RATIO_THRESHOLD = 0.005

    frame = img

    h,w,c = frame.shape

    # hsv色空間に変換
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # 色を抽出する
    rl_ex_img = cv2.inRange(hsv,Rl_LOW_COLOR,Rl_HIGH_COLOR)
    rh_ex_img = cv2.inRange(hsv,Rh_LOW_COLOR,Rh_HIGH_COLOR)
    g_ex_img = cv2.inRange(hsv,G_LOW_COLOR,G_HIGH_COLOR)
    b_ex_img = cv2.inRange(hsv,B_LOW_COLOR,B_HIGH_COLOR)

    r_ex_img = cv2.addWeighted(rl_ex_img, 1, rh_ex_img, 1, 0)

    cv2.namedWindow("rl_ex_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("rh_ex_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("r_ex_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("g_ex_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("b_ex_img", cv2.WINDOW_NORMAL)
    cv2.imshow("rl_ex_img", rl_ex_img)
    cv2.imshow("rh_ex_img", rh_ex_img)
    cv2.imshow("r_ex_img", r_ex_img)
    cv2.imshow("g_ex_img", g_ex_img)
    cv2.imshow("b_ex_img", b_ex_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((2, 5), np.uint8)
    r_closing = cv2.morphologyEx(r_ex_img, cv2.MORPH_CLOSE, kernel)
    g_closing = cv2.morphologyEx(g_ex_img, cv2.MORPH_CLOSE, kernel)
    b_closing = cv2.morphologyEx(b_ex_img, cv2.MORPH_CLOSE, kernel)

    cv2.namedWindow("r_closing", cv2.WINDOW_NORMAL)
    cv2.namedWindow("g_closing", cv2.WINDOW_NORMAL)
    cv2.namedWindow("b_closing", cv2.WINDOW_NORMAL)
    cv2.imshow("r_closing", r_closing)
    cv2.imshow("g_closing", g_closing)
    cv2.imshow("b_closing", b_closing)

    # 輪郭抽出
    # rl_contours,rl_hierarchy = cv2.findContours(rl_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # rh_contours,rh_hierarchy = cv2.findContours(rh_ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    r_contours,r_hierarchy = cv2.findContours(r_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    g_contours,g_hierarchy = cv2.findContours(g_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    b_contours,b_hierarchy = cv2.findContours(b_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_area = frame.shape[0] * frame.shape[1] * 1e-5
    tmp = frame.copy()
    h,w,c = frame.shape

    contours_list = [r_contours,g_contours,b_contours]
    colors = [(0,0,255),(0,255,0),(255,0,0)]

    for  contours, rect_color in zip(contours_list, colors):
        if len(contours) > 0:
            for i, contour in enumerate(contours):
                rect = cv2.boundingRect(contour)
                if rect[2] < 10 or rect[3] < 10:
                    continue
                if rect[3] > 30:
                    continue
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                if rect[0] < 100 or rect[0] > 1350 or rect[1] < 100 or rect[1] > 1000:
                    continue
                cv2.rectangle(tmp, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), rect_color, 2)
    return(tmp)

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print('Arguments are too short')
    else:
        # 画像取り込み
        img = cv2.imread(args[1])
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)

        # テキストサイズ調整
        img = adjust_edge(img)
        # cv2.namedWindow("adjust_edge", cv2.WINDOW_NORMAL)
        # cv2.imshow("adjust_edge", img)

        # # 彩度調整
        # img = cvt_saturation(img)
        # cv2.namedWindow("cvt_saturation", cv2.WINDOW_NORMAL)
        # cv2.imshow("cvt_saturation", img)

        # # 明度調整
        # img = cvt_brightness(img)
        # cv2.namedWindow("cvt_brightness", cv2.WINDOW_NORMAL)
        # cv2.imshow("cvt_brightness", img)

        # BGR画像に分離
        b_img, g_img ,r_img = extract_bgr(img)
        cv2.namedWindow("b_img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("g_img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("r_img", cv2.WINDOW_NORMAL)
        cv2.imshow("b_img", b_img)
        cv2.imshow("g_img", g_img)
        cv2.imshow("r_img", r_img)

        # ヒストグラムプロット
        # plot_hist(img)
        # plot_ratio(img)

        # # ヒストグラム均一化
        # img = clahe(img)
        # cv2.namedWindow("clahe", cv2.WINDOW_NORMAL)
        # cv2.imshow("clahe", img)

        # # マーカー抽出
        # img = get_position(img)
        # cv2.namedWindow("get_position", cv2.WINDOW_NORMAL)
        # cv2.imshow("get_position", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()