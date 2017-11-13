import cv2
import numpy as np
import itertools

import os

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def one_hot_it(labels,w,h):
    x = np.zeros([w,h,12])
    for i in range(w):
        for j in range(h):
            x[i,j,labels[i][j]]=1
    return x


# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = './CamVid/'
# data_shape = 360*480
data_shape = 224*224


def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        # data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        # this load cropped images
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])[136:,256:]),2))
        label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[136:,256:][:,:,0],224,224))
        print('.',end='')
    return np.array(data), np.array(label)



train_data, train_label = load_data("train")
train_label = np.reshape(train_label,(367,data_shape,12))
print(train_label.shape)
test_data, test_label = load_data("test")
test_label = np.reshape(test_label,(233,data_shape,12))

val_data, val_label = load_data("val")
val_label = np.reshape(val_label,(101,data_shape,12))


np.save("./train_data", train_data)
np.save("data/train_label", train_label)

np.save("data/test_data", test_data)
np.save("data/test_label", test_label)

np.save("data/val_data", val_data)
np.save("data/val_label", val_label)
