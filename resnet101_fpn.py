# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import os
import numpy as np
from keras.callbacks import ModelCheckpoint
import tifffile as tiff
from keras.layers import Input
from keras import layers
from keras.layers import Activation,Dropout,UpSampling2D,merge
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
import random
import datetime

FILE_2015 = '../data/2015.tif'
FILE_2017 = '../data/2017.tif'
label_2015 = '../data/2015label.tiff'
label_2017 = '../data/2017label.tiff'
#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
#WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet101(input_shape):

    # Determine proper input shape
    #input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,data_format=K.image_data_format(),include_top=include_top)

    inputs = Input(shape=input_shape)

    bn_axis = 3

    xk = Conv2D(64, (3,3), padding='same',name='convk')(inputs)
    x = ZeroPadding2D((3, 3))(xk)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    xa = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    xb = Activation('relu')(xa)
    xb = MaxPooling2D((2, 2), strides=(2, 2))(xb)

    x1 = conv_block(xb, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='c')

    x2 = conv_block(x1, 3, [128, 128, 512], stage=3, block='a')
    x2 = identity_block(x2, 3, [128, 128, 512], stage=3, block='b')
    x2 = identity_block(x2, 3, [128, 128, 512], stage=3, block='c')
    x2 = identity_block(x2, 3, [128, 128, 512], stage=3, block='d')

    x3 = conv_block(x2, 3, [256, 256, 1024], stage=4, block='a')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='b')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='c')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='d')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='e')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='f')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='g')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='h')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='i')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='j')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='k')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='l')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='m')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='n')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='o')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='p')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='q')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='r')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='s')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='t')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='u')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='v')
    x3 = identity_block(x3, 3, [256, 256, 1024], stage=4, block='w')

    x4 = conv_block(x3, 3, [512, 512, 2048], stage=5, block='a')
    x4 = identity_block(x4, 3, [512, 512, 2048], stage=5, block='b')
    x4 = identity_block(x4, 3, [512, 512, 2048], stage=5, block='c')

    #x5 = AveragePooling2D((7, 7), name='avg_pool')(x4)
    #x5 = Flatten()(x5)
    #x5 = Dense(classes, activation='softmax', name='fc1000')(x5)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # Create model.
    #model = Model(inputs, x5, name='resnet101') 
    
    #合并resnet101及FPN

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([x3,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([x2,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([x1,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([xa,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = merge([xk,up10], mode = 'concat', concat_axis = 3)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)    
    
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)


    model = Model(input = inputs, output = conv11,name='resnet101-fpn')
    #model = Model(input = inputs, output = x4,name='resnet101-fpn')

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])    


    return model

    
def images():
    
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    
    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
    
    im_label_2015 = tiff.imread(label_2015)
    
    im_label_2017 = tiff.imread(label_2017)
   
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
        
    train_x=[]
    train_y=[]

    for i in range(200): 
        # print(i)
        a=random.randint(0,1)
        #分成矩形框位置
        w=random.randint(0,5105)
        h=random.randint(0,15105)
        w1=w+256
        h1=h+256
        if w1<5106 and h1<15106:
            if a==0:
                aa=scale_percentile(im_2015[w:w1, h:h1, :3])
                bb=im_label_2015[w:w1, h:h1]
                bb=bb.reshape(256,256,1)
            if a==1:
                aa=scale_percentile(im_2017[w:w1, h:h1, :3])
                bb=im_label_2017[w:w1, h:h1]
                bb=bb.reshape(256,256,1)
            train_x.append(aa)
            train_y.append(bb)
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    return train_x,train_y
    
def yuce_pingjie(model):
    
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    
    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
   
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
        
    jg_2015=np.zeros([3000,15106],dtype=np.uint8)
    jg_2017=np.zeros([3000,15106],dtype=np.uint8)
    
    #2015yuce
    x_q=0
    y_q=0
    for i in range(100):
        if x_q<2743:
            y_q=0            
            for j in range(100):
                if y_q<14849:
                    aa=scale_percentile(im_2015[x_q:x_q+256,y_q:y_q+256,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range(len(bb.shape[0])):
                        for j in range(len(bb.shape[1])):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2015[x_q+s,y_q+j]=1
                    y_q+=256
                if y_q>=14849 and y_q<15105:
                    aa=scale_percentile(im_2015[x_q:x_q+256,-256:,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range(len(bb.shape[0])):
                        for j in range((15105-y_q)):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2015[x_q+s,y_q+j]=1
                    y_q+=256 
            x_q+=256
                          
        if x_q>=2743 and x_q<3000:
            y_q=0            
            for j in range(100):
                if y_q<14849:
                    aa=scale_percentile(im_2015[-256:,y_q:y_q+256,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range((2999-x_q)):
                        for j in range(bb.shape[1]):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2015[x_q+s,y_q+j]=1
                    y_q+=256
                if y_q>=14849 and y_q<15105:
                    aa=scale_percentile(im_2015[-256:,-256:,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range((2999-x_q)):
                        for j in range((15105-y_q)):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2015[x_q+s,y_q+j]=1
                    y_q+=256  
            x_q+=256
                          
    #2017yuce
    x_q=0
    y_q=0
    for i in range(100):
        if x_q<2743:
            y_q=0            
            for j in range(100):
                if y_q<14849:
                    aa=scale_percentile(im_2017[x_q:x_q+256,y_q:y_q+256,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range(len(bb.shape[0])):
                        for j in range(len(bb.shape[1])):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2017[x_q+s,y_q+j]=1
                    y_q+=256
                if y_q>=14849 and y_q<15105:
                    aa=scale_percentile(im_2017[x_q:x_q+256,-256:,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range(len(bb.shape[0])):
                        for j in range((15105-y_q)):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2017[x_q+s,y_q+j]=1
                    y_q+=256 
            x_q+=256
                                      
        if x_q>=2743 and x_q<3000:
            y_q=0            
            for j in range(100):
                if y_q<14849:
                    aa=scale_percentile(im_2017[-256:,y_q:y_q+256,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range((2999-x_q)):
                        for j in range(bb.shape[1]):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2017[x_q+s,y_q+j]=1
                    y_q+=256
                if y_q>=14849 and y_q<15105:
                    aa=scale_percentile(im_2017[-256:,-256:,:3])
                    aa=np.array([aa])
                    bb=model.predict(aa)
                    for s in range((2999-x_q)):
                        for j in range((15105-y_q)):
                            x=bb[s,j]
                            if x >=0.5:
                                jg_2017[x_q+s,y_q+j]=1
                    y_q+=256  
            x_q+=256
    #zuoca
    jg_2017_2015=jg_2017-jg_2015
    for i in range(jg_2017_2015.shape[0]):
        for j in range(jg_2017_2015.shape[0]):
            if jg_2017_2015[i,j] !=0 and jg_2017_2015[i,j] !=1:
                jg_2017_2015[i,j]=0
    
    return jg_2017_2015,jg_2015,jg_2017    


if __name__ == '__main__':
    
    print("get resnet_101_fpn model")
    model = ResNet101((256,256,3))
    #plot_model(model, to_file='H:\\kerasxx\\FCN\\model_resnet101_fpn.png',show_shapes=True)

    #plot_model(model, to_file='H:\\kerasxx\\FCN\\model_concat1.png',show_shapes=True)

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Fitting model...")
    for i in range(100):
        train_x,train_y=images()
        model.fit(train_x,train_y, batch_size=32, nb_epoch=10, verbose=1, shuffle=True, callbacks=[model_checkpoint])

    #利用模型得出结果
    #得出结果
    jg_2017_2015, jg_2015, jg_2017=yuce_pingjie(model)
    
    v=datetime.datetime.now()
    day=v.day
    month=v.month
    path='./result_%s_%s.tif'%(month,day)
    tiff.imsave(path, jg_2017_2015)
    tiff.imsave("./predict_2015.tif", jg_2015)
    tiff.imsave("./predict_2017.tif", jg_2017)
    
    
    
    
    
    
    
    