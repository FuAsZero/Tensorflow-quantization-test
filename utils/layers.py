import tensorflow as tf
import numpy as np

def print_info(x):
    FP32_input_size = tf.cast(tf.size(x), tf.float32)
    FP32_input_nonzero = tf.cast(tf.count_nonzero(x), tf.float32)
    FP32_nonzero_ratio = FP32_input_nonzero / FP32_input_size
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)
    s = tf.divide(vmax, 127.)
    x = tf.divide(x, s)
    x = tf.rint(x)
    INT8_input_nonzero = tf.cast(tf.count_nonzero(x), tf.float32)
    INT8_nonzero_ratio = INT8_input_nonzero / FP32_input_size
    return [1-FP32_nonzero_ratio, 1-INT8_nonzero_ratio]


def quantize(x, quant=False): #add quant=false to disable quant by default
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)

    if quant:
        print("INT8")
        s = tf.divide(vmax, 127.)
        x = tf.divide(x, s)
        x = tf.rint(x)
    else:
        print("FP32")
        s = vmax
    return x, s


def batch_norm(x, mean, variance, offset=None, scale=None):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-3)


def conv_2d(x, w, b=None, weight_scale=0., strides=1, padding='SAME', dilations=[1,1,1,1], activation=''):
    '''
    2D convolution with quantization (float32-->int8)
    '''
    # add print_out to print info
    print_out = print_info(x)

    # quantize input tensor
    quant = True
#    quant = False
    x, sx = quantize(x, quant)

    # Actually, convolution compute using float32,
    # because of tensorflow has not supported int8 conv op.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    # multiply scales
    s = sx * weight_scale
    x = x * s
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x, print_out #add print_out to print info


def depthwise_conv2d(x, w, b=None, strides=1, padding='SAME', activation=''):
    x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def separable_conv2d(x, dw, pw, dw_scale=0., pw_scale=0., strides=1, padding='SAME', activation=''):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.separable_conv2d(x, dw, pw, strides=[1, strides, strides, 1], padding=padding)
    # multiply scales
    x = x * sx * dw_scale * pw_scale
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def denselayer(x, w, b, weight_scale=0., activation=''):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.matmul(x, w)
    s = sx * weight_scale
    x = x * s
    x = tf.add(x, b)
    if activation == "relu":
        x = tf.nn.relu(x)
    return x


def maxpool_2d(x, k=2, s=2, padding='VALID'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


def avgpool_2d(x, k=2, s=1, padding='VALID'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],
                          padding=padding)
