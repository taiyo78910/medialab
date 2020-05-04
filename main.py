import sys
import cv2
from src import preprocess
from src import make

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        print('Arguments are too short')
    else:
        make.dir()
        img = cv2.imread(args[1])
        img = preprocess.adjust_edge(img)
        img = preprocess.clahe(img)
        img = preprocess.get_position(img)
        make.filename(img)
