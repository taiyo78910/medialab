import os
import glob

path = '/home/taiyoh/workplace/test/img/*'

dir_list = glob.glob(path)

for i in dir_list:
    n_list = i.split('/')
    head = n_list[2]
    dir1_list = glob.glob(i + '/*')
    for j in dir1_list:
        m_list = j.split('/')
        mid = m_list[3]
        file_list = glob.glob(j + '/*')
        no = 1
        for k in file_list:
            os.rename(k, j + '/' + head + mid + str(no).zfill(2) + '.jpg')
            no += 1