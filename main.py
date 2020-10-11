import sys
import cv2
import numpy as np
import datetime
from src import preprocess
from src import make

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print('Arguments are too short')
    else:
        cnt = 0
        # make.dir()
        img = cv2.imread(args[1])
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img",img)
        img = preprocess.adjust_edge(img)
        cv2.namedWindow("adjust_edge", cv2.WINDOW_NORMAL)
        cv2.imshow("adjust_edge",img)
        tmp = img.copy()
        img = preprocess.resize(img)
        img = preprocess.dilation(img)
        cv2.namedWindow("dilation", cv2.WINDOW_NORMAL)
        cv2.imshow("dilation",img)
        img = preprocess.cvt_saturation(img)
        cv2.namedWindow("cvt_saturation", cv2.WINDOW_NORMAL)
        cv2.imshow("cvt_saturation",img)
        img = preprocess.cvt_brightness(img)
        cv2.namedWindow("cvt_brightness", cv2.WINDOW_NORMAL)
        cv2.imshow("cvt_brightness",img)
        gamma_img = preprocess.gamma_correction(img)
        cv2.namedWindow("gamma_correction", cv2.WINDOW_NORMAL)
        cv2.imshow("gamma_correction",gamma_img)
        b_img, g_img ,r_img = preprocess.extract_bgr(img)
        gamma_b_img, gamma_g_img ,gamma_r_img = preprocess.extract_bgr(gamma_img)
        b_mask, g_mask, r_mask = preprocess.ext_color(b_img, g_img, r_img)
        gamma_b_mask, gamma_g_mask, gamma_r_mask = preprocess.ext_color(gamma_b_img, gamma_g_img, gamma_r_img)

        height, width= img.shape[:2]
        zeros = np.zeros((height, width), b_mask.dtype)
        B = cv2.merge((gamma_b_mask, zeros, zeros))
        G = cv2.merge((zeros, g_mask, zeros))
        R = cv2.merge((zeros, zeros, r_mask))
        print(img.dtype)
        print(b_img.dtype)

        cv2.namedWindow("B", cv2.WINDOW_NORMAL)
        cv2.imshow("B", B.astype(np.uint8))
        cv2.namedWindow("G", cv2.WINDOW_NORMAL)
        cv2.imshow("G", G.astype(np.uint8))
        cv2.namedWindow("R", cv2.WINDOW_NORMAL)
        cv2.imshow("R", R.astype(np.uint8))
        tmp = preprocess.get_mask_position(r_mask, tmp, cnt)
        cnt += 1
        tmp = preprocess.get_mask_position(g_mask, tmp, cnt)
        cnt += 1
        tmp = preprocess.get_mask_position(gamma_b_mask, tmp, cnt)
        # make.filename(tmp)
        # name = args[1].split('/')[4]
        # path = "/home/taiyoh/workplace/test/result/" + str(datetime.date.today()) + '/' + name.replace('.jpg', '') + '_result.jpg'
        # cv2.imwrite(path,tmp)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result",tmp)
        cv2.waitKey(0)

