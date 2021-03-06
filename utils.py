import tensorflow as tf
import scipy.io


def get_model_data(dir_path):
    """
    读取vgg模型的mat文件
    Parameters
    ----------
        dir_path: 存放模型的路径
    Returns
    -------
        data: 读取的模型文件
    """
    data = scipy.io.loadmat(dir_path)
    return data


def get_variable(weights, name):
    """
    定义tensorflow中的变量
    Parameters
    ----------
        weights: 需要定义的模型变量
        name: 变量的名字
    Return
    ------
        var: 经过初始化之后的定义好的变量
    """
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name = name, initializer = init,  shape = weights.shape)
    return var


def weight_variable(shape, stddev = 0.02, name=None):
    """
    通过标准差为0.02，均值为0的正态分布初始化卷积层变量
    Parameters
    ----------
        shape: 变量的维度
        stddev: 正态分布的标准差
        name: 变量的名字
    """
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    """
    通过常数值0初始化卷积层变量
    Parameters
    ----------
        shape: 变量的维度
        stddev: 正态分布的标准差
        name: 变量的名字
    """
    initial = tf.constant(0.0, shape = shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)




def conv2d_basic(x, W, bias):
    """
    对输入图像计算卷积层，步长为[1, 1, 1, 1]
    Parameters
    ----------
        x: 输入的图像
        W: 计算卷积的kernel
        bias: 卷积层的偏差项
    Returns
    -------
        经过卷积处理之后的feature map
    """
    # strides为四维的步长tensor
    # padding为SAME表示卷积核可以在图像边缘，输出的map和输入的图像大小则一致
    # 输入图像的tensor为四维，包括图像数量batch, 图像高度，图像宽度，图像通道数
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    """
        对输入图像计算卷积层，步长为[1, 2, 2, 1]
        Parameters
        ----------
            x: 输入的图像
            W: 计算卷积的kernel
            bias: 卷积层的偏差项
        Returns
        -------
            经过卷积处理之后的feature map
        """
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape = None, stride = 2):
    """
    实现反向卷积层
    Parameters
    ----------
        x: 需要反卷积处理的图像
        W: 反卷积的filter
        b: 反卷积的bias
        output_shape: 确定反卷积输出的维度
        stride: 反卷积的步长
    Returns
    -------
        反卷积的feature map
    """
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def max_pool_2x2(x):
    """
    计算2*2的最大池化层, 步长为2
    Parameters
    ----------
        x: 卷积层输出的feature map
    Returns
    -------
        池化层输出的feature map
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    """
    计算2*2的平均池化层, 步长为2
    Parameters
    ----------
        x: 卷积层输出的feature map
    Returns
    -------
        池化层输出的feature map
    """
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    """
    图片处理前减去均值，归一化处理
    Parameters
    ----------
        image: 输入的图像
        mean_pixel: 图像的均值
    """
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    """
        加上均值，显示图片
        Parameters
        ----------
            image: 输入的图像
            mean_pixel: 图像的均值
        """
    return image + mean_pixel



def add_to_regularization_and_summary(var):
    """
        通过summary记录监视变量，然后通过TensorBoard实现数据可视化
    Parameters
    ----------
        var: 需要记录的l2正则化损失函数
    """
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    """
        通过summary记录监视变量，然后通过TensorBoard实现数据可视化
    Parameters
    ----------
        var: 需要记录的激励函数变量
    """
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    """
    通过summary记录监视变量，然后通过TensorBoard实现数据可视化
    Parameters
    ----------
        grad: 变量的梯度值
        var: 通过summary记录监视变量
    """
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
        tf.summary.histogram(var.op.name + "/gradient", grad)