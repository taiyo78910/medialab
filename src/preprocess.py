import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from PIL import Image
import math

def adjust_edge(img):
    # リサイズ
    height, width = img.shape[:2]

    # magnification = 

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

def resize(img):
    height, width = img.shape[:2]
    magnification = float(1000/height)
    img = cv2.resize(img , (int(width*magnification), int(height*magnification)))

    return img

def gamma_correction(img):
    # img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # roi = [img.shape[3]/10, img.shape[3]*9/10, img.shape[2]/10, img.shape[2]*9/10]
    # s_roi = img_hsv[roi[1]: roi[3], roi[0]: roi[2]]
    # gamma =  math.log(np.mean(s_roi[:,:,(2)])) * (-3.5) + 20
    gamma = 1.8
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)

    query_img = cv2.LUT(img, gamma_cvt)
    return query_img

def restrict_color(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    brightness_thereshold_l = np.array([0, 0, 140])
    brightness_thereshold_h = np.array([179, 255, 255])
    # rl_therehold_l = np.array([0, 40 , 80])
    # rl_therehold_h = np.array([30, 80 , 110])
    # rh_therehold_l = np.array([150, 40 , 80])
    # rh_therehold_h = np.array([179, 80 , 110])
    # high_brightness_thereshold_l = np.array([0, 0, 250])
    # high_brightness_thereshold_h = np.array([179, 255, 255])

    brightness_mask = cv2.inRange(hsv,brightness_thereshold_l,brightness_thereshold_h)
    # rl_mask = cv2.inRange(hsv,rl_therehold_l,rl_therehold_h)
    # rh_mask = cv2.inRange(hsv,rh_therehold_l,rh_therehold_h)
    # high_brightness_mask = cv2.inRange(hsv,high_brightness_thereshold_l,high_brightness_thereshold_h)
    # red_mask = cv2.addWeighted(rl_mask, 1, rh_mask, 1, 0)

    # cv2.namedWindow("brightness_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("brightness_mask", brightness_mask)
    # cv2.namedWindow("high_brightness_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("high_brightness_mask", high_brightness_mask)

    brightness_mask = cv2.merge((brightness_mask, brightness_mask, brightness_mask))
    # red_mask = cv2.merge((red_mask, red_mask ,red_mask))
    # red_mask = cv2.bitwise_not(red_mask)

    # cv2.namedWindow("red_mask", cv2.WINDOW_NORMAL)
    # cv2.imshow("red_mask", red_mask)

    masked_img = cv2.bitwise_and(img, brightness_mask)
    # masked_img = cv2.bitwise_and(masked_img, red_mask)
    # masked_img = cv2.addWeighted(img, 1, brightness_mask, 1, 0)

    return masked_img

def restrict_dark(img):
    b_pix = img[:,:,0]
    g_pix = img[:,:,1]
    r_pix = img[:,:,2]
    pix = np.zeros(b_pix.shape[:2],np.int32)
    pix = pix + b_pix + g_pix + r_pix

    pix_mean = np.mean(pix)

    pix_mask = np.where((pix/pix_mean) > 0.7 ,255 ,0)

    mask = cv2.merge((pix_mask, pix_mask, pix_mask))
    mask = mask.astype(np.uint8)

    masked_img = cv2.bitwise_and(img, mask)

    cv2.namedWindow("masked_img", cv2.WINDOW_NORMAL)
    cv2.imshow("masked_img", masked_img)

    return masked_img

def restrict_saturation(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    thereshold_l = np.array([0, 0 , 200])
    thereshold_h = np.array([179, 255 , 255])
    mask = cv2.inRange(hsv,thereshold_l,thereshold_h)
    mask = cv2.merge((mask, mask, mask))
    mask = cv2.bitwise_not(mask)
    
    masked_img = cv2.bitwise_and(img, mask)

    return mask

def deshade(img):
    while(True):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thed = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        kernel = np.ones((8, 8), np.uint8)
        morphed = cv2.morphologyEx(thed, cv2.MORPH_CLOSE, kernel)

        inv_morphed = cv2.bitwise_not(morphed)

        conts, hierarchy = cv2.findContours(inv_morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros_like(img_gray)
        stopper = 0
        for index in range(len(conts)):
            if (cv2.contourArea(conts[index]) > 12000): # TODO: change to scheme
                cv2.drawContours(mask, conts, index, 255, -1)
                stopper+=1

        if stopper == 0:
            print("there aren't shade to remove. exit.")
            break

        x, y, shade_cropped = crop(mask, img)
        lighted_up_shade = light_up(shade_cropped)

        back = Image.fromarray(img[:,:,::-1])
        lighted_up_shade = cv2.cvtColor(lighted_up_shade, cv2.COLOR_BGRA2RGBA)
        target = Image.fromarray(lighted_up_shade)

        back.paste(target, (y, x), target)

        img = np.array(back)
    return img

def crop(mask, target):
    cropped = cv2.merge((mask, mask, mask))
    cropped[mask == 255] = target[mask == 255]
    alpha = np.ones_like(cropped) * 255
    cropped = cv2.merge((cropped, alpha[:, :, 1]))

    cropped[mask == 0] = (0, 0, 0, 0)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))

    return topx, topy, cropped[topx:bottomx+1, topy:bottomy+1]

def light_up(target, gamma=1.18):
    look_up_table = np.zeros((256, 1), dtype=np.uint8)

    for i in range(256):
        look_up_table[i] =  255 * pow(float(i) / 255, 1.0 / gamma)

    return cv2.LUT(target, look_up_table)

def dilation(img):
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def cvt_saturation(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    roi = [int(img.shape[0]*0.1), int(img.shape[0]*0.9), int(img.shape[1]*0.1), int(img.shape[1]*0.9)]
    s_roi = img_hsv[roi[0]: roi[1], roi[2]: roi[3]]
    s_rate = 20 / np.mean(s_roi[:,:,1])
    img_hsv[:,:,1] = img_hsv[:,:,1] * s_rate
    img = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
    return(img)

def cvt_brightness(img):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    roi = [int(img.shape[0]*0.1), int(img.shape[0]*0.9), int(img.shape[1]*0.1), int(img.shape[1]*0.9)]
    s_roi = img_hsv[roi[0]: roi[1], roi[2]: roi[3]]
    b_rate = 220 / np.mean(s_roi[:,:,(2)])
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

def ext_color(b_img, g_img, r_img):
    r = 1e-10
    b_pix = b_img[:,:,0]
    g_pix = g_img[:,:,1]
    r_pix = r_img[:,:,2]

    pix = np.zeros(b_pix.shape[:2],np.int32)
    pix = pix + b_pix + g_pix + r_pix
    b_mean = b_pix / (pix + r) 
    g_mean = g_pix / (pix + r) 
    r_mean = r_pix / (pix + r) 

    # bw_mask = np.where((b_mean > 0.3) & (g_mean > 0.3) & (r_mean > 0.3), False, True)

    b_mean_mask = np.where((b_mean >0.35) & (pix > 300) ,255 ,0)
    g_mean_mask = np.where((g_mean >0.35) & (b_mean < 0.35) & (pix > 300) ,255 ,0)
    r_mean_mask = np.where((r_mean >0.37) & (pix > 300) ,255 ,0)
    # b_mean_mask = np.where((b_mean_mask == 255) & (bw_mask == True) ,255 ,0)
    # g_mean_mask = np.where((g_mean_mask == 255) & (bw_mask == True) ,255 ,0)
    # r_mean_mask = np.where((r_mean_mask == 255) & (bw_mask == True) ,255 ,0)

    # height, width= b_mean_mask.shape[:2]
    # zeros = np.zeros((height, width), b_mean_mask.dtype)
    # b_mean_mask = cv2.merge((b_mean_mask, zeros, zeros))
    # zeros = np.zeros((height, width), g_mean_mask.dtype)
    # g_mean_mask = cv2.merge((zeros, g_mean_mask, zeros))
    # zeros = np.zeros((height, width), r_mean_mask.dtype)
    # r_mean_mask = cv2.merge((zeros, zeros, r_mean_mask))
    
    # b_mean_mask = b_mean_mask.astype(np.uint8)
    # g_mean_mask = g_mean_mask.astype(np.uint8)
    # r_mean_mask = r_mean_mask.astype(np.uint8)
    
    # kernel = np.ones((1,0),np.uint8)
    # mean_mask = cv2.erode(mean_mask,kernel,iterations = 1)

    return b_mean_mask, g_mean_mask, r_mean_mask

def merge_mask(b_mask, g_mask, r_mask):
    mean_mask = cv2.merge((b_mask, g_mask, r_mask))
    mean_mask = mean_mask.astype(np.uint8)
    return mean_mask

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
    ratio = []
    color = ('b','g','r')
    for i,col in enumerate(color):
        hist_list.append(cv2.calcHist([img],[i],None,[256],[0,256]))
    
    for j,col in enumerate(color):
        ratio[j] = (hist_list[j] / (hist_list[0] + hist_list[1] + hist_list[2] + r))
    #     plt.plot(ratio,color = col)
    #     plt.xlim([0,256])
    # plt.show()
    return ratio
    
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

    # cv2.namedWindow("rl_ex_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("rh_ex_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("r_ex_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("g_ex_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("b_ex_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("rl_ex_img", rl_ex_img)
    # cv2.imshow("rh_ex_img", rh_ex_img)
    # cv2.imshow("r_ex_img", r_ex_img)
    # cv2.imshow("g_ex_img", g_ex_img)
    # cv2.imshow("b_ex_img", b_ex_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((2, 5), np.uint8)
    r_closing = cv2.morphologyEx(r_ex_img, cv2.MORPH_CLOSE, kernel)
    g_closing = cv2.morphologyEx(g_ex_img, cv2.MORPH_CLOSE, kernel)
    b_closing = cv2.morphologyEx(b_ex_img, cv2.MORPH_CLOSE, kernel)

    # cv2.namedWindow("r_closing", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("g_closing", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("b_closing", cv2.WINDOW_NORMAL)
    # cv2.imshow("r_closing", r_closing)
    # cv2.imshow("g_closing", g_closing)
    # cv2.imshow("b_closing", b_closing)

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

def get_marker(img, tmp):
    # BGRでの色抽出
    b_lower = np.array([130, 0, 0])
    b_upper = np.array([255, 0, 0])
    b_mask = cv2.inRange(img, b_lower, b_upper)
    g_lower = np.array([0, 130, 0])
    g_upper = np.array([0, 255, 0])
    g_mask = cv2.inRange(img, g_lower, g_upper)
    r_lower = np.array([0, 0, 130])
    r_upper = np.array([0, 0, 255])
    r_mask = cv2.inRange(img, r_lower, r_upper)
    # result = cv2.bitwise_and(img, image, mask=img_mask)

    kernel = np.ones((1, 10), np.uint8)
    b_closing = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)
    g_closing = cv2.morphologyEx(g_mask, cv2.MORPH_CLOSE, kernel)
    r_closing = cv2.morphologyEx(r_mask, cv2.MORPH_CLOSE, kernel)

    # 輪郭抽出
    r_contours,r_hierarchy = cv2.findContours(r_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    g_contours,g_hierarchy = cv2.findContours(g_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    b_contours,b_hierarchy = cv2.findContours(b_closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.namedWindow("b_closing", cv2.WINDOW_NORMAL)
    cv2.namedWindow("g_closing", cv2.WINDOW_NORMAL)
    cv2.namedWindow("r_closing", cv2.WINDOW_NORMAL)
    cv2.imshow("b_closing", b_closing)
    cv2.imshow("g_closing", g_closing)
    cv2.imshow("r_closing", r_closing)

    min_area = img.shape[0] * img.shape[1] * 1e-5
    h,w,c = img.shape

    height, width = tmp.shape[:2]
    magnification = float(height/1000)

    roi = [int(img.shape[0]*0.1), int(img.shape[0]*0.9), int(img.shape[1]*0.05), int(img.shape[1]*0.9)]

    contours_list = [r_contours,g_contours,b_contours]
    colors = [(0,0,255),(0,255,0),(255,0,0)]

    for  contours, rect_color in zip(contours_list, colors):
        if len(contours) > 0:
            for i, contour in enumerate(contours):
                rect = cv2.boundingRect(contour)
                if rect[2] < (10*magnification) or rect[3] < (10*magnification):
                    continue
                if rect[3] > (30*magnification):
                    continue
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                if rect[0] < roi[2] or rect[0] > roi[3] or rect[1] < roi[0] or rect[1] > roi[1]:
                    continue
                cv2.rectangle(tmp, (int(rect[0]*magnification), int(rect[1] * magnification)), (int((rect[0]+rect[2])*magnification), int((rect[1]+rect[3])*magnification)), rect_color, int(2*magnification))
    return(tmp)

def get_mask_position(mask, tmp, cnt):
    # mask = cv2.merge((mask, mask, mask))
    mask = mask.astype(np.uint8)

    kernel = np.ones((3, 10), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    color_list = ["r", "g", "b"]
    # cv2.namedWindow(color_list[cnt], cv2.WINDOW_NORMAL)
    # cv2.imshow(color_list[cnt], closing)


    min_area = mask.shape[0] * mask.shape[1] * 1e-5
    h,w = mask.shape

    height, width = tmp.shape[:2]
    magnification = float(height/1000)

    roi = [int(mask.shape[0]*0.1), int(mask.shape[0]*0.9), int(mask.shape[1]*0.05), int(mask.shape[1]*0.9)]

    colors = [(0,0,255),(0,255,0),(255,0,0)]

    for i, contour in enumerate(contours):
        rect = cv2.boundingRect(contour)
        if rect[2] < 5 or rect[3] < 3:
            continue
        if rect[3] > 30:
            continue
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if rect[0] < roi[2] or rect[0] > roi[3] or rect[1] < roi[0] or rect[1] > roi[1]:
            continue
        cv2.rectangle(tmp, (int(rect[0]*magnification), int(rect[1] * magnification)), (int((rect[0]+rect[2])*magnification), int((rect[1]+rect[3])*magnification)), colors[cnt], int(2*magnification))
    return(tmp)

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print('Arguments are too short')
    else:
        cnt = 0
        # 画像取り込み
        img = cv2.imread(args[1])
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        # cv2.imwrite("input.jpg",img)

        # トリミング
        img = adjust_edge(img)
        tmp = img.copy()
        cv2.namedWindow("adjust_edge", cv2.WINDOW_NORMAL)
        cv2.imshow("adjust_edge", img)
        # cv2.imwrite("adjust_edge.jpg",img)

        # 画像サイズ調整
        img = resize(img)

        # 文字削除
        # img = dilation(img)
        # # cv2.namedWindow("dilation", cv2.WINDOW_NORMAL)
        # # cv2.imshow("dilation", img)
        # # cv2.imwrite("dilation.jpg",img)
        
        # # 彩度調整
        # img = cvt_saturation(img)
        # # cv2.imwrite("cvt_saturation.jpg",img)
        # # cv2.namedWindow("cvt_saturation", cv2.WINDOW_NORMAL)
        # # cv2.imshow("cvt_saturation", img)

        # # 明度調整
        # img = cvt_brightness(img)
        # # cv2.imwrite("cvt_brightness.jpg",img)
        # # cv2.namedWindow("cvt_brightness", cv2.WINDOW_NORMAL)
        # # cv2.imshow("cvt_brightness", img)

        # # ガンマ補正
        # gamma_img = gamma_correction(img)
        # cv2.namedWindow("gamma_correction", cv2.WINDOW_NORMAL)
        # cv2.imshow("gamma_correction", gamma_img)
        # cv2.imwrite("gamma_correction.jpg",gamma_img)

        # 影除去
        # img = deshade(img)
        # cv2.imwrite("deshade.jpg",p_img)
        # cv2.namedWindow("deshade", cv2.WINDOW_NORMAL)
        # cv2.imshow("deshade", img)
        
        # ヒストグラム均一化
        # p_img = clahe(img)
        # cv2.imwrite("clahe.jpg",p_img)
        # cv2.namedWindow("clahe", cv2.WINDOW_NORMAL)
        # cv2.imshow("clahe", img)

        # ヒストグラム均一化(ガンマ補正済)
        # gamma_img = clahe(gamma_img)
        # cv2.namedWindow("gamma_clahe", cv2.WINDOW_NORMAL)
        # cv2.imshow("gamma_clahe", gamma_img)

        # 彩度の低い部分を除去
        # img = restrict_saturation(img)
        # cv2.imwrite("restrict_saturation.jpg",p_img)
        # cv2.namedWindow("restrict_saturation", cv2.WINDOW_NORMAL)
        # cv2.imshow("restrict_saturation", img)
        
        # 黒い部分の除去
        # img = restrict_color(img)
        # cv2.imwrite("restrict_color.jpg",p_img)
        # cv2.namedWindow("restrict_color", cv2.WINDOW_NORMAL)
        # cv2.imshow("restrict_color", img)

        # 色の濃い部分を除去
        # img = restrict_dark(img)
        # # cv2.imwrite("restrict_dark.jpg",p_img)
        # # cv2.namedWindow("restrict_dark", cv2.WINDOW_NORMAL)
        # # cv2.imshow("restrict_dark", img)

        # # BGR画像に分離
        # b_img, g_img ,r_img = extract_bgr(img)
        # # cv2.namedWindow("b_img", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("g_img", cv2.WINDOW_NORMAL)
        # # cv2.namedWindow("r_img", cv2.WINDOW_NORMAL)
        # # cv2.imshow("b_img", b_img)
        # # cv2.imshow("g_img", g_img)
        # # cv2.imshow("r_img", r_img)

        # # BGR画像に分離(ガンマ補正済)
        # gamma_b_img, gamma_g_img ,gamma_r_img = extract_bgr(gamma_img)
        # cv2.namedWindow("gamma_b_img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("gamma_g_img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("gamma_r_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("gamma_b_img", gamma_b_img)
        # cv2.imshow("gamma_g_img", gamma_g_img)
        # cv2.imshow("gamma_r_img", gamma_r_img)

        # ヒストグラムプロット
        plot_hist(img)
        # ratio = plot_ratio(img)
        # cv2.namedWindow("b_ratio", cv2.WINDOW_NORMAL)
        # cv2.imshow("b_ratio", ratio[0])
        # cv2.namedWindow("g_ratio", cv2.WINDOW_NORMAL)
        # cv2.imshow("g_ratio", ratio[1])
        # cv2.namedWindow("r_ratio", cv2.WINDOW_NORMAL)
        # cv2.imshow("r_ratio", ratio[2])

        # # 色抽出
        # b_mask, g_mask, r_mask = ext_color(b_img, g_img, r_img)
        # # cv2.namedWindow("b_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("b_mask", b_mask)
        # # cv2.namedWindow("g_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("g_mask", g_mask)
        # # cv2.namedWindow("r_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("r_mask", r_mask)

        # # 色抽出(ガンマ補正済)
        # gamma_b_mask, gamma_g_mask, gamma_r_mask = ext_color(gamma_b_img, gamma_g_img, gamma_r_img)
        # # cv2.namedWindow("gamma_b_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("gamma_b_mask", gamma_b_mask)
        # # cv2.namedWindow("gamma_g_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("gamma_g_mask", gamma_g_mask)
        # # cv2.namedWindow("gamma_r_mask", cv2.WINDOW_NORMAL)
        # # cv2.imshow("gamma_r_mask", gamma_r_mask)

        # # # マーカー抽出
        # # img = get_position(img)
        # # cv2.namedWindow("get_position", cv2.WINDOW_NORMAL)
        # # cv2.imshow("get_position", img)

        # # マスク合成
        # img = merge_mask(gamma_b_mask, g_mask, r_mask)
        # cv2.imwrite("mask.jpg",img)
        # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask", img)

        # # マーカー抽出ver.2
        # # img = get_marker(img, tmp)
        # # cv2.namedWindow("get_marker", cv2.WINDOW_NORMAL)
        # # cv2.imshow("get_marker", img)

        # # マーカー抽出ver.3
        # tmp = get_mask_position(r_mask, tmp, cnt)
        # cnt += 1
        # tmp = get_mask_position(g_mask, tmp, cnt)
        # cnt += 1
        # tmp = get_mask_position(gamma_b_mask, tmp, cnt)
        # cv2.imwrite("result.jpg",tmp)
        # # cv2.namedWindow("get_mask_position", cv2.WINDOW_NORMAL)
        # # cv2.imshow("get_mask_position", tmp)

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()