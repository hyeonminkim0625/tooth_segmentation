from glob import glob
import pickle
import random
import cv2
import numpy as np
from tqdm import tqdm

data = glob('/data/maicon_train/y/*png')

class_list_list = []
data_list = []

data_dict = {}

for i, d in enumerate(tqdm(data)):
    img = cv2.imread(d,0)
    h,w = img.shape
    prev_img = img[:,:w//2]
    after_img = img[:,w//2:]
    prev_img[prev_img==2] = 10
    after_img[after_img==2] = 10
    prev_img[prev_img==3] = 100
    after_img[after_img==3] = 100

    class_list = np.unique(prev_img+after_img)
    if 11 in class_list or 110 in class_list or 101 in class_list or 111 in class_list:
        continue

    class_list = np.sort(class_list)
    class_list = class_list.tolist()
    class_list = [str(i) for i in class_list]

    class_list_list.append(''.join(class_list))
    data_list.append(d)

class_list_list = np.array(class_list_list)
data_list = np.array(data_list)
for i in np.unique(class_list_list):
    data_dict[i] = data_list[class_list_list==i]


np.save('data_list.npy',data_dict)
    