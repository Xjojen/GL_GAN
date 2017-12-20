# Version: 1.0 2017.12.12
# Author: Jojen

import os
import numpy as np

from model import GLGAN

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# normal beta1 is 0.9
# flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
# we use the 128x128 face images set to train
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "dataset", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("maskType", "center", "choice your own mask type")
# mask type ['random', 'center', left', 'full', 'grid', 'lowres'], default is center
# 传递所有命令行所接收的参数给FLAGS
FLAGS = flags.FLAGS

# 判断模型和样本文件夹是否存在，若不存则创建一个新的
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
with tf.Session(config=config) as sess:
    glgan = GLGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=True, checkpoint_dir=FLAGS.checkpoint_dir)
    # 假如is_crop=False, 那么这个自带裁切会截取图片中央的部分；所以对于训练人脸数据应该自己对图片进行预处理
    glgan.train(FLAGS)
