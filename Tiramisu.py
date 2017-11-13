import keras.models as models
import tifffile as tiff
import random
import datetime
import os
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras import callbacks
import math


FILE_2015_first = './data/quickbird2015.tif'
FILE_2017_first = './data/quickbird2017.tif'
FILE_2015_second = './data/2015.tif'
FILE_2017_second = './data/2017.tif'
FILE_2015_PRE = './data/fu_2015.tif'
FILE_2017_PRE = './data/fu_2017.tif'
label_2015_first = './data/2015label.tiff'
label_2017_first = './data/2017label.tiff'
label_2015_second = './data/relabel_2015.tif'
label_2017_second = './data/relabel_2017.tif'
import numpy as np
import json
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

K.set_image_dim_ordering('tf')

# 指定第3块GPU可用
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print('start train')
class_weights = np.zeros((224, 224, 2))
class_weights[:, :, 0] += 0.5
class_weights[:, :, 1] += 0.5
# load the data
def one_hot_it(labels,w,h):
    x = np.zeros([w,h,2])
    for i in range(w):
        for j in range(h):
            x[i,j,int(labels[i][j])]=1
    return x


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def create_samples(image, label):
    images = []
    labels = []

    height = image.shape[0]
    weight = image.shape[1]
    IMAGE_SIZE = 224
    height_step = 224
    width_step = 224
    for i in range(int(((height - IMAGE_SIZE) / height_step))):
        for j in range(int((weight - IMAGE_SIZE) / width_step)):
            b = random.randint(1, 3)
            h = i * height_step
            h1 = h + IMAGE_SIZE
            w = j * width_step
            w1 = w + IMAGE_SIZE
            if b == 1:
                aa = scale_percentile(image[h:h1, w:w1, :3])
                aa = np.rot90(aa)
                bb = label[h:h1, w:w1]
                bb = one_hot_it(bb, 224, 224)
                bb = np.rot90(bb).reshape(224 * 224, 2)
            if b == 2:
                aa = scale_percentile(image[h:h1, w:w1, :3])
                aa = np.rot90(aa, 2)
                bb = label[h:h1, w:w1]
                bb = one_hot_it(bb, 224, 224)
                bb = np.rot90(bb, 2).reshape(224 * 224, 2)
            if b == 3:
                aa = scale_percentile(image[h:h1, w:w1, :3])
                aa = np.rot90(aa, 3)
                bb = label[h:h1, w:w1]
                bb = one_hot_it(bb, 224, 224)
                bb = np.rot90(bb, 3).reshape(224 * 224, 2)
            images.append(aa)
            labels.append(bb)
    print('image count: ',len(images))
    return images, labels


def images():
    print('start get image')
    im_2015_first = tiff.imread(FILE_2015_first).transpose([1, 2, 0])
    im_2015_second = tiff.imread(FILE_2015_second).transpose([1, 2, 0])
    im_2017_first = tiff.imread(FILE_2017_first).transpose([1, 2, 0])
    im_2017_second = tiff.imread(FILE_2017_second).transpose([1, 2, 0])
    im_label_2015_first = tiff.imread(label_2015_first)
    im_label_2015_second = tiff.imread(label_2015_second)
    im_label_2017_first = tiff.imread(label_2017_first)
    im_label_2017_second = tiff.imread(label_2017_second)
    train_y = []
    train_x = []
    image_label_total = [[im_2015_first,im_label_2015_first], [im_2015_second, im_label_2015_second], [im_2017_first, im_label_2017_first], [im_2017_second, im_label_2017_second]]
    for element in image_label_total:
        images, labels = create_samples(element[0],element[1])
        for image in images:
            train_x.append(image)
        for label in labels:
            train_y.append(label)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

# load the model:
with open('./tiramisu_fc_dense67_model_12_func.json') as model_file:
    tiramisu = models.model_from_json(model_file.read())
print('load model')
# section 4.1 from the paper

from keras.callbacks import LearningRateScheduler


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)

# tiramisu.load_weights("weights/prop_tiramisu_weights_67_12_func_10-e5_decay.best.hdf5")

def yuce_pingjie(model):
    im_2015 = tiff.imread(FILE_2015_PRE).transpose([1, 2, 0])

    im_2017 = tiff.imread(FILE_2017_PRE).transpose([1, 2, 0])

    def scale_percentile(matrix):
        w, h, d = matrix.shape
        matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
        # Get 2nd and 98th percentile
        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        matrix = np.reshape(matrix, [w, h, d])
        matrix = matrix.clip(0, 1)
        return matrix

    jp_2015 = np.zeros([4000, 15106, 2], dtype=np.float32)
    jp_2017 = np.zeros([4000, 15106, 2], dtype=np.float32)
    IMAGE_SIZE = 224
    height_step = 224
    width_step = 224
    image_height = im_2015.shape[0]
    image_weight = im_2015.shape[1]
    num_matrix = np.zeros([image_height, image_weight, 2])
    for i in range(int(((image_height - IMAGE_SIZE) / height_step))):
        for j in range(int((image_weight - IMAGE_SIZE) / width_step)):
            hs = i * height_step
            he = hs + IMAGE_SIZE
            ws = j * width_step
            we = ws + IMAGE_SIZE
            test_2015 = scale_percentile(im_2015[hs:he, ws:we, :3])
            test_2017 = scale_percentile(im_2017[hs:he, ws:we, :3])
            result_2015 = model.predict(np.array([test_2015])).reshape(224,224,2)
            result_2017 = model.predict(np.array([test_2017])).reshape(224,224,2)
            jp_2015[hs:he, ws:we, :] += result_2015
            jp_2017[hs:he, ws:we, :] += result_2017
    print("predict finish")
    return jp_2015, jp_2017


optimizer = RMSprop(lr=0.0001, decay=0.000001)
# optimizer = SGD(lr=0.01)
#optimizer = Adam(lr=1e-4, decay=0.995)

tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# learning schedule callback
#lrate = LearningRateScheduler(step_decay)

# checkpoint 278
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)
# ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

filepath = "weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2,
                             save_best_only=True, save_weights_only=False, mode='max')

callbacks_list = [checkpoint]

nb_epoch = 100
batch_size = 2

# Fit the model
for i in range(2):
    train_x,train_y=images()
    history = tiramisu.fit(train_x,train_y, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks_list, verbose=1, shuffle=True)  # validation_split=0.33
tiramisu.save_weights('./weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))
#tiramisu.load_weights('./weights/prop_tiramisu_weights_67_12_func_10-e7_decay10.hdf5')
jp_2015, jp_2017 = yuce_pingjie(tiramisu)

tiff.imsave("./predict_2015.tif", jp_2015)
tiff.imsave("./predict_2017.tif", jp_2017)
print('2015',jp_2015[:,:,1].sum())
print('2017',jp_2017[:,:,1].sum())

