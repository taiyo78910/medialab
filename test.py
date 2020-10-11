import cv2
from matplotlib import pyplot as plt

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