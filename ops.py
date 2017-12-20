import tensorflow as tf


class BatchNorm(object):
    """
    This class used to normalize the batch
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_nrom"):
        """

        :param epsilon: float, added to variance to avoid dividing by zero
        :param momentum: float, decay for moving average
        :param name: optional scope for variable_scope
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """

        :param x: tensor with dimension of [batch_size, ...]
        :param train: whether or not the layer is in training model
        :return: A tensor representing the output of the operation
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    """

    :param input_: 4D-tensor, shape is [batch, height, width, channels]
    :param output_dim: int, output channel
    :param k_h: int, kernel height
    :param k_w: int, kernel width
    :param d_h: int, the second dimensional(height) stride of input_
    :param d_w: int, the third dimensional(width) stride of input_
    :param stddev: float, standard deviation of w
    :param name: string, variable scope
    :return: 4D-tensor, the result of convolution network
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def dilated_conv2d(input_, output_dim, k_h=3, k_w=3, dilation=2, stddev=0.02, name="dilated_conv2d"):
    """

    :param input_: 4D-tensor, shape is [batch, height, width, channels]
    :param output_dim: int, output channel
    :param k_h: int, kernel high
    :param k_w: int, kernel width
    :param dilation: int, dilation rate, enlarge the receptive field
    :param stddev: float, standard deviation of w
    :param name: string, variable scope
    :return: positive int, the dilation arugment
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.atrous_conv2d(input_, w, rate=dilation, padding="SAME")
        conv = tf.nn.bias_add(conv, biases)
        return conv


def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2,
                     name="conv2d_transpose", with_w=False):
    """

    :param input_: 4D-tensor, shape is [batch, height, width, channels]
    :param output_shape: A 1-D tensor, representing the output shape of the deconvolution op
    :param k_h: int, kernel height
    :param k_w: int, kernel width
    :param d_h: int, the second dimensional(height) stride of input_
    :param d_w: int, the third dimensional(width) stride of input_
    :param stddev: float, standard deviation
    :param name: string, variable scope
    :param with_w: boolean, whether return w & bias or not
    :return:
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            # print(output_shape.get_shape())
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        except AttributeError:
            print("Your tensroflow version is too old, please update to 0.70+")
            return False

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    """

    :param x: 4D-tensor, shape is [batch, height, width, channels]
    :param leak: float, denote the degree or change
    :param name: string, variable scope
    :return: 4D-tensor, shape is [batch, height, width, channels], each number is after activation
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
