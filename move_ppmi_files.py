# Jesse Liu

import os
import shutil

dir = r'C:\Users\jesse\OneDrive\Desktop\Research\PD\PD_all'
dest = r'C:\Users\jesse\OneDrive\Desktop\Research\PD\raw_files\PD'

files1 = os.listdir(dir)

for i in range(len(files1)):
    files2 = os.listdir(dir + '\\' + files1[i])
    for j in range(len(files2)):
        files3 = os.listdir(dir + '\\' + files1[i] + '\\' + files2[j])
        for k in range(len(files3)):
            files4 = os.listdir(dir + '\\' + files1[i] + '\\' + files2[j] + '\\' + files3[k])
            for m in range(len(files4)):
                files5 = os.listdir(dir + '\\' + files1[i] + '\\' + files2[j] + '\\' + files3[k] + '\\' + files4[m])
                for n in range(len(files5)):
                    image = dir + '\\' + files1[i] + '\\' + files2[j] + '\\' + files3[k] + '\\' + files4[m] + '\\' + files5[n]
                    shutil.move(image, dest)
