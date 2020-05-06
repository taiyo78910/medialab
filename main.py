import sys
import cv2
from src import preprocess
from src import make

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print('Arguments are too short')
    else:
        cnt = 0
        make.dir()
        img = cv2.imread(args[1])
        img = preprocess.adjust_edge(img)
        tmp = img.copy()
        img = preprocess.resize(img)
        img = preprocess.dilation(img)
        img = preprocess.cvt_saturation(img)
        img = preprocess.cvt_brightness(img)
        gamma_img = preprocess.gamma_correction(img)
        b_img, g_img ,r_img = preprocess.extract_bgr(img)
        gamma_b_img, gamma_g_img ,gamma_r_img = preprocess.extract_bgr(gamma_img)
        b_mask, g_mask, r_mask = preprocess.ext_color(b_img, g_img, r_img)
        gamma_b_mask, gamma_g_mask, gamma_r_mask = preprocess.ext_color(gamma_b_img, gamma_g_img, gamma_r_img)
        tmp = preprocess.get_mask_position(r_mask, tmp, cnt)
        cnt += 1
        tmp = preprocess.get_mask_position(g_mask, tmp, cnt)
        cnt += 1
        tmp = preprocess.get_mask_position(gamma_b_mask, tmp, cnt)
        make.filename(tmp)
