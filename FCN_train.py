import tensorflow as tf
import numpy as np
import utils
import tifffile as tiff
import os
import random


FILE_2015 = '../data/2015.tif'
FILE_2017 = '../data/2017.tif'
label_2015 = '../data/2015label.tiff'
label_2017 = '../data/2017label.tiff'
FILE_2015_PRE = '../data/quickbird2015.tif'
FILE_2017_PRE = '../data/quickbird2017.tif'
train_Path = '../training_pic/'

# 定义命令行参数，然后tensorflow会解析这些flags, 如果不给定参数，则使用默认值
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "./imagenet-vgg-verydeep-19.mat", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ predict")
MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224

def vgg_net(weights, image):
    """
    首先通过vgg模型初始化权值
    Parameters
    ----------
        weights: vgg模型的权重
        image: 训练的样本图片
    Returns
    -------
        net: vgg模型初始化之后的模型
    """
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # tensorflow和mat的卷积核格式不一样，需要做个transpose变换
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name = name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    return net


def segmentation(image, keep_prob):
    """
    图像语义分割模型定义
    Parameters
    ----------
        image: 输入图像，每个通道的像素值为0到255
        keep_prob: 防止过拟合的dropout参数
    Returns
    -------

    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir)
    # vgg模型的权重值
    weights = np.squeeze(model_data['layers'])
    # 计算图片像素值的均值, 然后对图像加上均值
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    processed_image = utils.process_image(image, mean_pixel)
    # 共享变量名空间-segmentation
    with tf.variable_scope("segmentation"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)
        # 全连接层用卷积层来代替
        W6 = utils.weight_variable([7, 7, 512, 4096], name = "W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        # 随机去掉一些神经元防止过拟合
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name = "b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape = deconv_shape3, stride = 8)
        # 预测结果层
        annotation_pred = tf.argmax(conv_t3, dimension = 3, name = "prediction")

    return tf.expand_dims(annotation_pred, dim = 3), conv_t3


def train(loss_val, var_list):
    """
    定义采用那种算法的优化器，然后计算loss函数的梯度值，并加入到summary中
    Parameters
    ----------
        loss_val: 计算的loss值
        var_list: 需要计算梯度的变量
    """
    # 采用Adam算法的优化器
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # 以下可以合并成一步minimize
    grads = optimizer.compute_gradients(loss_val, var_list = var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def shuffle_namelist(Path):
    """
    该函数是用于将所有样本图像的名字打乱
    Parameters
    ----------
        Path:训练样本根目录
    Returns
    -------
        name_list: 打乱之后的样本名字列表
    """
    name_list = list(set([name.split('_')[0] for name in os.listdir(Path)]))
    random.shuffle(name_list)
    return name_list



def read_2_namelist(name_list, batch_size, start_index):
    """
    该函数用于获取每次迭代所需的正负样本图像名字列表
    Parameters
    ----------
        name_list: 所有正负样本的名字列表
        batch_size: 每次训练迭代需要的样本数
        start_index: 开始的编号
    Returns
    -------
        image_batch: 包含正负样本名字的数组
        label_batch: 包含标签的样本名字和数组
    """
    image_batch = []
    label_batch = []
    if start_index + batch_size > len(name_list):
        random.shuffle(name_list)
        start_index = start_index + batch_size - len(name_list)
    for name in name_list[start_index:start_index + batch_size]:
        image_batch.append(tiff.imread(train_Path + name + '_samples.tif'))
        label_batch.append(tiff.imread(train_Path + name + '_label.tif'))
    return np.array(image_batch), np.array(label_batch), (start_index + batch_size) % len(name_list)


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


def predict(sess, pred_annotation, height_step = 224, width_step = 224):
    image = tf.placeholder(tf.float32, shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    im_2015 = tiff.imread(FILE_2015_PRE).transpose([1, 2, 0])

    im_2017 = tiff.imread(FILE_2017_PRE).transpose([1, 2, 0])
    image_weight = im_2015.shape[1]
    image_height = im_2015.shape[0]
    pred_2015 = np.zeros([image_height, image_weight])
    pred_2017 = np.zeros([image_height, image_weight])
    num_matrix = np.zeros([image_height, image_weight])
    for i in range(int(((image_height - IMAGE_SIZE) / height_step))):
        for j in range(int((image_weight - IMAGE_SIZE) / width_step)):
            hs = i * height_step
            he = hs + IMAGE_SIZE
            ws = j * width_step
            we = ws + IMAGE_SIZE
            test_2015 = scale_percentile(im_2015[hs:he, ws:we, :])
            test_2017 = scale_percentile(im_2017[hs:he, ws:we, :])
            result_2015 = sess.run(pred_annotation, feed_dict={image: test_2015, keep_probability: 1.0}).flatten().reshape(224, 224)
            result_2017 = sess.run(pred_annotation, feed_dict={image: test_2017, keep_probability: 1.0}).flatten().reshape(224, 224)
            pred_2015[hs:he, ws:we] += result_2015
            pred_2017[hs:he, ws:we] += result_2017
            num_matrix[hs:he, ws:we] += np.zeros([IMAGE_SIZE, IMAGE_SIZE])
    print("predict finish")
    num_matrix = num_matrix.clip(min=1)
    pred_2015 = pred_2015 / num_matrix
    pred_2017 = pred_2017 / num_matrix
    threshold = 0.5
    pred_2015[pred_2015 >= threshold] = 1
    pred_2015[pred_2015 < threshold] = 0
    pred_2017[pred_2017 >= threshold] = 1
    pred_2017[pred_2017 < threshold] = 0
    print('pred_2015=', type(pred_2015), pred_2015.dtype, pred_2015.shape, pred_2015.sum())
    print('pred_2017=', type(pred_2017), pred_2017.dtype, pred_2017.shape, pred_2017.sum())
    tiff.imsave('./pred_2015.tif', pred_2015)
    tiff.imsave('./pred_2017.tif', pred_2017)


def main(argv = None):
    """
    tensorflow 中的app.run会先解析命令行参数flag，然后执行main函数
    """
    # 定义存放图像，标签和过拟合的数据结构
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = segmentation(image, keep_probability)
    # 添加监控信息，可以通过tensorboard查看
    tf.summary.image("input_image", image, max_outputs = 2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs = 2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs = 2)
    # 还没看懂loss函数是怎么计算的，reduce_mean是求平均值，squeeze为去掉维度是1
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))

    tf.summary.scalar("entropy", loss)
    # 获取在训练中的变量列表
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    name_list = shuffle_namelist(train_Path)
    start_index = 0
    batch_size = FLAGS.batch_size
    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            train_images, train_annotations, start_index = read_2_namelist(name_list, batch_size, start_index)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)
            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "FCN_model.ckpt", itr)
    if FLAGS.mode == "predict":
        saver.restore(sess, FLAGS.logs_dir + "FCN_model.ckpt")
        predict(sess, pred_annotation)



if __name__ == "__main__":
    tf.app.run()
