import datetime
import os
import re
import shutil
import cv2

def dir():
    dirpath = "/home/taiyoh/workplace/test/result/"+str(datetime.date.today())
    if os.path.exists(dirpath):
        return()
    else:
        os.mkdir(dirpath)
    return()

def filename(img):
    cnt = 0
    while True:
        addpara = '(' + '{0:3d}'.format(cnt) + ')'
        filepath = "/home/taiyoh/workplace/test/result/"+str(datetime.date.today())+"/"+str(datetime.date.today())+addpara+".jpg"
        if os.path.exists(filepath):  
            cnt += 1
            continue
        else:
            break
    cv2.imwrite(filepath,img)


if __name__ == '__main__':
    make_dir()