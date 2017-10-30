import numpy as np
import tifffile as tiff
import os
import shutil

root_dir='../training_pic/'
FILE_2015 = '../data/2015_3channel.tif'
FILE_2017 = '../data/2017_3channel.tif'
label_2015 = '../data/2015label.tiff'
label_2017 = '../data/2017label.tiff'
IMAGE_SIZE = 224


def __init__():
    """
    该函数为初始化操作，创建图像保存路径
    """
    if os.path.exists(root_dir):
        print (root_dir+' exits,remove all')
        shutil.rmtree(root_dir)
    print (root_dir+' not exits,will be create')
    os.mkdir(root_dir)


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


def create_samples():
    __init__()
    im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
    im_label_2015 = tiff.imread(label_2015)
    im_label_2017 = tiff.imread(label_2017)
    image_weight = im_2015.shape[1]
    image_height = im_2015.shape[0]
    print('each_pic_size =', IMAGE_SIZE * IMAGE_SIZE, '=', IMAGE_SIZE, '*', IMAGE_SIZE)
    height_step = 28
    width_step = 28
    num = 0
    for i in range(int(((image_height - IMAGE_SIZE) / height_step))):
        for j in range(int((image_weight - IMAGE_SIZE) / width_step)):
            num += 2
            hs = i * height_step
            he = hs + IMAGE_SIZE
            ws = j * width_step
            we = ws + IMAGE_SIZE
            aa = scale_percentile(im_2015[hs:he, ws:we, :])
            bb = im_label_2015[hs:he, ws:we]
            bb = bb.reshape(224, 224, 1)
            aa_2 = scale_percentile(im_2017[hs:he, ws:we, :])
            bb_2 = im_label_2017[hs:he, ws:we]
            bb_2 = bb_2.reshape(224, 224, 1)
            base_path_2015 = root_dir + '%d-%d&%d-%d-2015_' % (hs, he, ws, we)
            tiff.imsave(base_path_2015 + 'samples.tif', aa)
            tiff.imsave(base_path_2015 + 'label.tif', bb)
            base_path_2017 = root_dir + '%d-%d&%d-%d-2017_' % (hs, he, ws, we)
            tiff.imsave(base_path_2017 + 'samples.tif', aa_2)
            tiff.imsave(base_path_2017 + 'label.tif', bb_2)
    print('Samples num =', num)

if __name__ == "__main__":
    create_samples()