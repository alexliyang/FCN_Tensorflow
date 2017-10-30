import tensorflow as tf
from models import alexnet_fcn_voc, fcn8, res
import numpy as np
import tifffile as tiff
import random, os


FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "initial learning rate for training")
tf.app.flags.DEFINE_string("checkpoint_path", 'res_34_64/', "path to checkpoint_dir")
tf.app.flags.DEFINE_float('decay_factor', 0.01, "lr decay factor")
tf.app.flags.DEFINE_integer('decay_steps', 200, "lr decay step")
tf.app.flags.DEFINE_integer("max_steps", 10000, "the max step of train")
tf.app.flags.DEFINE_integer("num_class", 2, "the num of classes")
tf.app.flags.DEFINE_integer("batch_size", 5, "the size of batch")

train_Path = '../training_pic/'
IMAGE_SIZE = 224

def loss(logits, labels):
    """
    定义损失函数
    Parameters
    ----------
        logits: softmax的输入层
        labels: 标签数据
    Returns
    -------
        返回交叉熵函数
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cross_entropy_mean)
    # return cross_entropy_mean
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')


def train_op(total_loss, global_step):
    """
    定义训练的参数
    Parameters
    ----------
        total_loss: 全局损失函数
        global_step: 全局步长
    """
    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_factor,
                                    staircase=True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 控制上下文依赖关系
    with tf.control_dependencies([apply_gradient_op]):
        train_op_ = tf.no_op(name='train')
    return train_op_





def init_config(log_device=False, gpu_memory_fraction=0.8):
    """
    设置gpu的使用参数
    :param log_device:
    :param gpu_memory_fraction:
    :return:
    """
    gs = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    config_ = tf.ConfigProto(log_device_placement=log_device, gpu_options=gs)
    return config_

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


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        label_ = tf.placeholder("float", [None])
        image_ = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
        shape_ = tf.placeholder("int32", [4])
        # logits = alexnet_fcn_voc(image_, shape_)
        logits = res(image_, shape_, 5)
        logits = tf.reshape(logits, (-1, FLAGS.num_class))
        loss_ = loss(logits, label_)
        train_op_ = train_op(loss_, global_step)
        sess = tf.Session(config=init_config())
        init = tf.initialize_all_variables()
        sess.run(init)
        start_index = 0
        name_list = shuffle_namelist(train_Path)
        for step in range(FLAGS.max_steps):
            data, label, start_index = train_images, train_annotations, start_index = read_2_namelist(name_list, FLAGS.batch_size, start_index)
            # image_.set_shape([None,None,None,3])
            # image_.set_shape(data.shape)
            logits_ = sess.run(logits, feed_dict={image_: data, shape_: data.shape})
            loss_value, _, logits_tmp = sess.run([loss_, train_op_, tf.argmax(tf.nn.softmax(logits), dimension=1)],
                                                 feed_dict={label_: label, image_: data, shape_: data.shape})
            if step % 100 == 0:
                print("Step: %d, Train_loss:%g" % (step, loss_value))



if __name__ == '__main__':
    train()