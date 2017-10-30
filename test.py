import tifffile as tiff
import matplotlib.pyplot
import tensorflow as tf
def variable_with_weight_loss(shape, stddev, wl):
    """
    该函数按照正态分布初始化权重
    Parameters
    ----------
        shape: 权值矩阵大小
        stddev: 正态分布标准差
        wl: 是否采用L2正则化
    Returns
    -------
        var: 随机生成的权值初始值
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
if __name__ == "__main__":
    out_label_num = 24 * 24
    image_holder = tf.placeholder(tf.float32, [1, 24, 24, 8])
    weight1 = variable_with_weight_loss(shape=[5, 5, 8, 64], stddev=5e-2, wl=0)
    kernel1 = tf.nn.conv2d(image_holder, weight1, strides=[1, 1, 1, 1], padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    print("conv1: ", conv1.shape)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("pool1: ", pool1.shape)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print("norm1: ", norm1.shape)
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0)
    kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    print("conv2: ", conv2.shape)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print("norm2: ", norm2.shape)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print("pool2: ", pool2.shape)
    reshape = tf.reshape(pool2, [1, -1])
    print("reshape: ", reshape.shape)
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
    print("local3: ", local3.shape)
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
    print("local4: ", local4.shape)
    weight5 = variable_with_weight_loss(shape=[192, out_label_num], stddev=1 / 192.0, wl=0.0)
    bias5 = tf.Variable(tf.constant(0.0, shape=[out_label_num]))
    # logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)
    logits = tf.matmul(local4, weight5) + bias5
    print("logits: ", logits.shape)
    predict = tf.nn.sigmoid(logits)
    print("predict: ", predict.shape)