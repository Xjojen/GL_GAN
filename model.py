# Version: 1.0 2017.12.12
# Author: Jojen

from __future__ import division
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from ops import BatchNorm, conv2d, dilated_conv2d, conv2d_transpose, lrelu
import itertools
from glob import glob
import os
import numpy as np
from utils import get_image, save_images
import time
from random import randint

SUPPORTED_EXTENSTIONS = ["png", "jpg", "jpeg"]


def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSTIONS))


class GLGAN(object):
    def __init__(self, sess, image_size=64, is_crop=True, batch_size=64, sample_size=64, checkpoint_dir=None,
                 c_dim=3):
        """

        :param sess: Tensorflow session
        :param image_size: int, the size of each image
        :param is_crop: boolean, images is cropped or not
        :param batch_size: int, the size of batch
        :param sample_size: int, the size of samples which will be show as images
        :param checkpoint_dir: string, the path of trained model
        :param c_dim: int, dimension of images channel
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.image_size = image_size
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.checkpoint_dir = checkpoint_dir
        self.c_dim = c_dim

        # batch normalization: deals with poor initialization helps gradient flow; i means discriminator layers number
        self.global_d_bns = [BatchNorm(name='global_d_bn{}'.format(i,)) for i in range(5)]
        self.local_d_bns = [BatchNorm(name='local_d_bn{}'.format(i, )) for i in range(4)]

        # i means generator layers number
        self.g_bns = [BatchNorm(name='g_bn{}'.format(i,)) for i in range(15)]

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # the third dimensional of image shape is RGB + binary channel
        self.image_shape = [image_size, image_size, self.c_dim]
        # self.mask_shape = [None, None, self.c_dim]
        self.real_images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.masked_images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='masked_images')
        self.masks_pos = tf.placeholder(tf.int32, [self.batch_size, 4], name='masks_pos')
        # self.local_images = tf.placeholder(tf.float32, [None] + self.mask_shape, name='local_images')

        # 生成修补后的完整图片
        self.G = self.generator(self.masked_images)

        # self.local_real_images = tf.placeholder(tf.float32, [None, 40, 40, 3], name='local_real_images')
        self.local_real_images = []
        for i in range(batch_size):
            self.local_real_images.append(tf.slice(self.real_images[i], [self.masks_pos[i, 0], self.masks_pos[i, 1], 0],
                                                   [40, 40, 3]))
        self.local_real_images = tf.stack(self.local_real_images)
        # print(self.local_real_images.get_shape())
        # 真实图片送入判别器
        self.D, self.D_logits = self.discriminator(self.real_images, self.local_real_images)

        # 生成图片送入判别器; if reuse: 用另外一套变量，防止变量名字冲突
        self.local_generated_images = []
        for i in range(batch_size):
            self.local_generated_images.append(tf.slice(self.G[i], [self.masks_pos[i, 0], self.masks_pos[i, 1], 0],
                                                        [40, 40, 3]))
        self.local_generated_images = tf.stack(self.local_generated_images)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.local_generated_images, reuse=True)

        # 以下三句表示保存三个变量到tensor board直方图观测
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D)
        self.G_sum = tf.summary.histogram("G", self.G)

        # 针对真实图片时的loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                                  labels=tf.ones_like(self.D)))
        # 针对生成图片的loss
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                                  labels=tf.zeros_like(self.D)))
        # 针对生成网络的loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                             labels=tf.ones_like(self.D_)))

        # 输出标量统计结果到tensor board中
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # 输出标量统计结果到tensor board中
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # max_to_keep: maximum number of recent checkpoints to keep
        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion
        # self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        # self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(
        #     tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        # self.contextual_loss += tf.reduce_sum(tf.contrib.layers.flatten(
        #     tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))),
        #     1)
        self.model_name = "GLGAN.model"

    def generator(self, images):
        """
        This is the completion network
        :param images: 4D-tensor, shape is [batch, height, width, channels]
        :return: 4D-tensor, shape is [batch, height, width, channels], completed images tensor
        """
        with tf.variable_scope("generator"):
            g_h0 = tf.nn.relu(conv2d(images, 64, 5, 5, 1, 1, name='g_h0'))

            g_h1 = tf.nn.relu(self.g_bns[0](conv2d(g_h0, 128, 3, 3, 2, 2, name='g_h1')))
            g_h2 = tf.nn.relu(self.g_bns[1](conv2d(g_h1, 128, 3, 3, 1, 1, name='g_h2')))

            g_h3 = tf.nn.relu(self.g_bns[2](conv2d(g_h2, 256, 3, 3, 2, 2, name='g_h3')))
            g_h4 = tf.nn.relu(self.g_bns[3](conv2d(g_h3, 256, 3, 3, 1, 1, name='g_h4')))
            g_h5 = tf.nn.relu(self.g_bns[4](conv2d(g_h4, 256, 3, 3, 1, 1, name='g_h5')))
            g_h6 = tf.nn.relu(self.g_bns[5](dilated_conv2d(g_h5, 256, 3, 3, 2, name='g_h6')))
            g_h7 = tf.nn.relu(self.g_bns[6](dilated_conv2d(g_h6, 256, 3, 3, 4, name='g_h7')))
            g_h8 = tf.nn.relu(self.g_bns[7](dilated_conv2d(g_h7, 256, 3, 3, 8, name='g_h8')))
            g_h9 = tf.nn.relu(self.g_bns[8](dilated_conv2d(g_h8, 256, 3, 3, 16, name='g_h9')))
            g_h10 = tf.nn.relu(self.g_bns[9](conv2d(g_h9, 256, 3, 3, 1, 1, name='g_h10')))
            g_h11 = tf.nn.relu(self.g_bns[10](conv2d(g_h10, 256, 3, 3, 1, 1, name='g_h11')))

            g_h12 = tf.nn.relu(self.g_bns[11](
                conv2d_transpose(g_h11, [self.batch_size, int(self.image_size / 2),
                                         int(self.image_size / 2), 128], 4, 4, 2, 2, name='g_h12')))
            g_h13 = tf.nn.relu(self.g_bns[12](conv2d(g_h12, 128, 3, 3, 1, 1, name='g_h13')))

            g_h14 = tf.nn.relu(
                self.g_bns[13](conv2d_transpose(g_h13, [self.batch_size, self.image_size, self.image_size, 64],
                                                4, 4, 2, 2, name='g_h14')))
            g_h15 = tf.nn.relu(self.g_bns[14](conv2d(g_h14, 32, 3, 3, 1, 1, name='g_h15')))
            g_h16 = tf.nn.sigmoid(conv2d(g_h15, 3, 3, 3, 1, 1, name='g_h16'))
            return g_h16

    def discriminator(self, images, local_images, reuse=False):
        """

        :param images: 4D-tensor, shape is [batch, height, width, channels]
        :param local_images: 4D-tensor, shape is [batch, mask_max_height, mask_max_width, channels]
        :param reuse: bool, used to prevent conflict of  variable name
        :return:
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            gd_h0 = lrelu(conv2d(images, 64, name='d_gd_h0_conv'))
            gd_h1 = lrelu(self.global_d_bns[0](conv2d(gd_h0, 128, name='d_gd_h1_conv')))
            gd_h2 = lrelu(self.global_d_bns[1](conv2d(gd_h1, 256, name='d_gd_h2_conv')))
            gd_h3 = lrelu(self.global_d_bns[2](conv2d(gd_h2, 512, name='d_gd_h3_conv')))
            gd_h4 = lrelu(self.global_d_bns[3](conv2d(gd_h3, 512, name='d_gd_h4_conv')))
            gd_h5 = lrelu(self.global_d_bns[4](conv2d(gd_h4, 512, name='d_gd_h5_conv')))

            # gd_h shape is [?, 1, 1, 1024)
            gd_h5_shape = [int(i) for i in gd_h5.get_shape()]
            gd_h = fully_connected(tf.reshape(gd_h5, [gd_h5_shape[0], gd_h5_shape[1]*gd_h5_shape[2]*gd_h5_shape[3]]),
                                   1024, scope='global_loss')

            ld_h0 = lrelu(conv2d(local_images, 64, name="d_ld_h0_conv"))
            ld_h1 = lrelu(self.local_d_bns[0](conv2d(ld_h0, 128, name="d_ld_h1_conv")))
            ld_h2 = lrelu(self.local_d_bns[1](conv2d(ld_h1, 256, name="d_ld_h2_conv")))
            ld_h3 = lrelu(self.local_d_bns[2](conv2d(ld_h2, 512, name='d_ld_h3_conv')))
            ld_h4 = lrelu(self.local_d_bns[3](conv2d(ld_h3, 512, name="d_ld_h4_conv")))

            ld_h4_shape = [int(i) for i in ld_h4.get_shape()]
            ld_h = fully_connected(tf.reshape(ld_h4, [ld_h4_shape[0], ld_h4_shape[1]*ld_h4_shape[2]*ld_h4_shape[3]]),
                                   1024, scope='local_loss')
            sum_loss = fully_connected(tf.concat([gd_h, ld_h], 1), 1, scope='d_loss')
            return tf.nn.sigmoid(sum_loss), sum_loss

    def train(self, config):
        data = dataset_files(config.dataset)
        np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdadeltaOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdadeltaOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge([self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # 创建一个file writer来向硬盘写入数据
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        # train begin
        if self.load(self.checkpoint_dir):
            print("""

        ======
        An existing model was found in the checkpoint directory.
        If you just cloned this repository, it's a model for faces
        trained on the CelebA dataset for 20 epochs.
        If you want to train a new model from scratch,
        delete the checkpoint directory or specify a different
        --checkpoint_dir argument.
        ======

        """)
        else:
            print("""

        ======
        An existing model was not found in the checkpoint directory.
        Initializing a new one.
        ======

        """)
        for epoch in range(config.epoch):
            data = dataset_files(config.dataset)
            # 计算有多少组数据
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                # batch is float ndarray with dimension of [batch_size, image_size, image_size, 3],
                # each element value within [-1, 1]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                # mean pixel value
                mean_pixels = np.mean(batch_images, axis=0)

                # generate random masks
                masks_pos = np.zeros((self.batch_size, 4))
                masked_images = batch_images.copy()
                for i in range(batch_images.shape[0]):
                    x = randint(0, 23)
                    y = randint(0, 23)
                    x_len = randint(20, 40)
                    y_len = randint(20, 40)
                    masks_pos[i] = [x, y, x_len, y_len]
                    masked_images[i][x:x+x_len][y:y+y_len][:] = mean_pixels[x:x+x_len][y:y+y_len][:]
                # masks.reshape(batch_images.shape[0], batch_images.shape[1], batch_images.shape[2], 1)
                # masks is a binary layers, takes the value 1 inside regions to be  filled-in and 0 elsewhere

                # batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
                    self.real_images: batch_images, self.masked_images: masked_images,
                    self.masks_pos: masks_pos, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_images: batch_images,
                                                                                 self.masked_images: masked_images,
                                                                                 self.masks_pos: masks_pos,
                                                                                 self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_images: batch_images,
                                                                                 self.masked_images: masked_images,
                                                                                 self.masks_pos: masks_pos,
                                                                                 self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                err_d_fake = self.d_loss_fake.eval({self.real_images: batch_images, self.masked_images: masked_images,
                                                    self.masks_pos: masks_pos, self.is_training: False})
                err_d_real = self.d_loss_real.eval({self.real_images: batch_images, self.masked_images: masked_images,
                                                   self.masks_pos: masks_pos, self.is_training: False})
                err_g = self.g_loss.eval({self.real_images: batch_images, self.masked_images: masked_images,
                                          self.masks_pos: masks_pos, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, err_d_fake+err_d_real, err_g))

                if np.mod(counter, 100) == 1:
                    # mean pixel value
                    sample_mean_pixels = np.mean(sample_images, axis=0)

                    # generate random masks
                    sample_masks_pos = np.zeros((sample_images.shape[0], 4))
                    sample_masked_images = sample_images.copy()
                    for i in range(sample_images.shape[0]):
                        x = randint(0, 23)
                        y = randint(0, 23)
                        x_len = randint(20, 40)
                        y_len = randint(20, 40)
                        sample_masks_pos[i] = [x, y, x_len, y_len]
                        sample_masked_images[i][x:x + x_len][y:y + y_len][:] = sample_mean_pixels[x:x + x_len][y:y + y_len][:]
                    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss],
                                                            feed_dict={self.real_images: sample_images,
                                                                       self.masked_images: sample_masked_images,
                                                                       self.masks_pos: sample_masks_pos,
                                                                       self.is_training: False})
                    save_images(samples, [8, 8], './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

