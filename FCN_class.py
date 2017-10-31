import numpy as np
import os
import tensorflow as tf
import tifffile as tiff
import random

import tf_utils
from FCN import create_fcn


class TensorFCN(object):
    def __init__(self,
                 classes = 2,
                 logs_dir = "./logs",
                 checkpoint = None,
                 train_path = "../training_pic"):
        """
        :param classes: number of classes for classification
        :param logs_dir: directory for logs
        :param checkpoint: a CheckpointState from get_checkpoint_state
        """
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')

        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.prediction, self.logits = create_fcn(
            self.image, self.keep_prob, classes)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss_op = self._loss()
        self.train_op = self._training(global_step)

        self.checkpoint = checkpoint
        self.train_path = train_path

    def _training(self, global_step):
        """
        Setup the training phase with Adam
        :param global_step: global step of training
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads = optimizer.compute_gradients(self.loss_op)
        for grad, var in grads:
            tf_utils.add_gradient_summary(grad, var, collections=['train'])
        return optimizer.apply_gradients(grads, global_step=global_step)

    def _loss(self):
        """
        Setup the loss function
        """
        return tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=self.annotation))

    def _setup_supervisor(self):
        """
        Setup the summary writer and variables
        """
        saver = tf.train.Saver(max_to_keep=20)
        sv = tf.train.Supervisor(
            logdir=self.logs_dir,
            save_summaries_secs=0,
            save_model_secs=0,
            saver=saver)

        # Restore checkpoint if given
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            print('Checkpoint found, loading model')
            with sv.managed_session() as sess:
                sv.saver.restore(sess, self.checkpoint.model_checkpoint_path)

        return sv

    def shuffle_namelist(self):
        """
        该函数是用于将所有样本图像的名字打乱
        Parameters
        ----------
            Path:训练样本根目录
        Returns
        -------
            name_list: 打乱之后的样本名字列表
        """
        name_list = list(set([name.split('_')[0] for name in os.listdir(self.train_path)]))
        random.shuffle(name_list)
        return name_list

    def read_2_namelist(self, name_list, batch_size, start_index):
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
            image_batch.append(tiff.imread(self.train_path + name + '_samples.tif'))
            label_batch.append(tiff.imread(self.train_path + name + '_label.tif'))
        return np.array(image_batch), np.array(label_batch), (start_index + batch_size) % len(name_list)

    def train(self,
              batch_size = 12,
              lr = 1e-5,
              keep_prob = 0.5,
              train_freq = 10,
              val_freq = 0,
              save_freq = 2000,
              max_steps = 10000):
        """
        :param lr: initial learning rate
        :param keep_prob: 1 - dropout
        :param train_freq: trace train_loss every train_freq iterations
        :param val_freq: trace val_loss every val_freq iterations
        :param save_freq: save model every save_freq iterations
        :param max_steps: max steps to perform
        """
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])
        tf.summary.scalar('train_loss', self.loss_op, collections=['train'])

        summ_train = tf.summary.merge_all(key='train')
        summ_val = tf.Summary()
        summ_val.value.add(tag='val_loss', simple_value=0)

        sv = self._setup_supervisor()
        name_list = self.shuffle_namelist()
        start_index = 0
        with sv.managed_session() as sess:
            print('Starting training...')
            while not sv.should_stop():

                images, anns, start_index = self.read_2_namelist(name_list, batch_size, start_index)
                # Transform to match NN inputs
                images = images.astype(np.float32) / 255.
                anns = np.int32(anns)
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.lr: lr,
                    self.keep_prob: keep_prob
                }
                sess.run(self.train_op, feed_dict=feed)

                step = sess.run(sv.global_step)

                if (step == max_steps) or ((step % train_freq) == 0):
                    loss, summary = sess.run(
                        [self.loss_op, summ_train],
                        feed_dict=feed)
                    sv.summary_computed(sess, summary, step)
                    print('Step %d\tTrain_loss: %g' % (step, loss))

                if (step == max_steps) or ((save_freq > 0) and
                                                   (step % save_freq) == 0):
                    # Save model
                    sv.saver.save(sess, os.path.join(self.logs_dir, 'model.ckpt'), step)
                    print('Step %d\tModel saved.' % step)
                    # TODO Save train & set dataset state

                if step == max_steps:
                    break

    def test(self, height_step = 224, width_step = 224):
        """
        Run on images of any size without their groundtruth
        :param files: list of absolute paths to images
        """
        IMAGE_SIZE = 224

        FILE_2015_PRE = '../data/quick_2015_3channels.tif'
        FILE_2017_PRE = '../data/quick_2017_3channels.tif'

        sv = self._setup_supervisor()

        with sv.managed_session() as sess:
            im_2015 = tiff.imread(FILE_2015_PRE)

            im_2017 = tiff.imread(FILE_2017_PRE)
            # pad image to the nearest multiple of 32
            dy, dx = tf_utils.get_pad(im_2015, mul=32)
            im_2015 = tf_utils.pad(im_2015, dy, dx)
            # batch size = 1
            im_2015 = np.expand_dims(im_2015, axis=0)
            im_2015 = im_2015.astype(np.float32) / 255.
            # pad image to the nearest multiple of 32
            dy1, dx1 = tf_utils.get_pad(im_2017, mul=32)
            im_2017 = tf_utils.pad(im_2017, dy1, dx1)
            # batch size = 1
            im_2017 = np.expand_dims(im_2017, axis=0)
            im_2017 = im_2017.astype(np.float32) / 255.
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
                    test_2015 = im_2015[hs:he, ws:we, :]
                    test_2017 = im_2017[hs:he, ws:we, :]
                    feed = {self.image: test_2015, self.keep_prob: 1.0}
                    result_2015 = sess.run(self.prediction, feed_dict=feed)
                    feed2 = {self.image: test_2017, self.keep_prob: 1.0}
                    result_2017 = sess.run(self.prediction, feed_dict=feed2)
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


