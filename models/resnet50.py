from utils.layers import *
import tensorflow as tf
import numpy as np


def quantize(weights, quant=False): #add quant=false to disable quant by default
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)

    if quant:
        total_size = np.size(weights)  #
        FP32_nonzero = np.count_nonzero(weights) #
        FP32_zero_ratio = 1 - FP32_nonzero / total_size #
        s = vmax / 127.
        qweights = weights / s
        qweights = np.round(qweights)
        qweights = qweights.astype(np.int8)
        INT8_nonzero = np.count_nonzero(qweights) #
        INT8_zero_ratio = 1 - INT8_nonzero / total_size #
        print("shape: ", np.shape(weights), "size: ", total_size) #
        print("FP32_zero_ratio: %.4f, INT8_zero_ratio: %.4f" % (FP32_zero_ratio, INT8_zero_ratio)) #
    else:
        s = vmax
        qweights = weights
    return qweights, s


def get_weights_biases_scale(weights, weight_name, bias_name='bbb', quant=False): #change quant to false to disable quant
    w = weights[weight_name]
    if quant:
#        w, s = quantize(w)
        w, s = quantize(w, quant)
        w = tf.constant(w, dtype=tf.float32)
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 0.
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s



def get_bn_param(weights, mean, std, beta, gamma):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    gamma = tf.constant(weights[gamma], dtype=tf.float32)
    return mean, std, beta, gamma


def identity_block(out_dict, input_dict, inputs, weights, stage, block, quant=False):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_params = ['_running_mean:0', '_running_std:0', '_beta:0', '_gamma:0']
    conv_wb = ['_W:0', '_b:0']
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
#    w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1])
    w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1], quant)

#    x = conv_2d(inputs, w, b, s)
    input_dict[conv] = inputs
    x, out_dict[conv] = conv_2d(inputs, w, b, s) 
    bn = bn_name_base + conv_names[0]
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)

    for i in range(1, 3):
#pragma unroll
        conv = conv_name_base + conv_names[i]
#        w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1])
        w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1], quant)

#        x = conv_2d(x, w, b, s)
        input_dict[conv] = x
        x, out_dict[conv] = conv_2d(x, w, b, s)
        bn = bn_name_base + conv_names[i]
        mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                              bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
        x = batch_norm(x, mean, std, beta, gamma)
        if i < 2:
            x = tf.nn.relu(x)
    x = tf.add(x, inputs)
    return tf.nn.relu(x)



def conv_block(out_dict, input_dict, inputs, weights, stage, block, strides=2, quant=False):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_params = ['_running_mean:0', '_running_std:0', '_beta:0', '_gamma:0']
    conv_wb = ['_W:0', '_b:0']
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
#    w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1])
    w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1], quant)

#    x = conv_2d(inputs, w, b, s, strides=strides)
    input_dict[conv] = inputs
    x, out_dict[conv] = conv_2d(inputs, w, b, s, strides=strides) #add print_out to print info
    bn = bn_name_base + conv_names[0]
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)

    for i in range(1, 3):
#pragma unroll
        conv = conv_name_base + conv_names[i]
#        w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1])
        w, b, s = get_weights_biases_scale(weights, conv + conv_wb[0], conv + conv_wb[1], quant)

#        x = conv_2d(x, w, b, s)
        input_dict[conv] = x
        x, out_dict[conv] = conv_2d(x, w, b, s) #add print_out to print info
        bn = bn_name_base + conv_names[i]
        mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                              bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
        x = batch_norm(x, mean, std, beta, gamma)
        if i < 2:
            x = tf.nn.relu(x)

    # shortcut
#    w, b, s = get_weights_biases_scale(weights, conv_name_base + '1_W:0', conv_name_base + '1_b:0')
    w, b, s = get_weights_biases_scale(weights, conv_name_base + '1_W:0', conv_name_base + '1_b:0', quant)
#    shortcut = conv_2d(inputs, w, b, s, strides=strides)
    input_dict[conv] = inputs
    shortcut, out_dict[conv] = conv_2d(inputs, w, b, s, strides=strides) #add print_out to print info
    bn = bn_name_base + '1'
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0],
                                          bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    shortcut = batch_norm(shortcut, mean, std, beta, gamma)
    x = tf.add(x, shortcut)
    return tf.nn.relu(x)



def ResNet50(x, weights, out_dict={}, input_dict={}):
    # init convolution
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    w, b, s = get_weights_biases_scale(weights, 'conv1_W:0', 'conv1_b:0')
#    x = conv_2d(x, w, b, s, strides=2)
#input_dict skip conv1 layer, since it has only 3 channels
    x, out_dict['conv1'] = conv_2d(x, w, b, s, strides=2)
    mean, std, beta, gamma = get_bn_param(weights, 'bn_conv1_running_mean:0',
                                          'bn_conv1_running_std:0', 'bn_conv1_beta:0', 'bn_conv1_gamma:0')
    x = batch_norm(x, mean, std, beta, gamma)
    x = tf.nn.relu(x)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    quant = True
#    quant = False
    x = conv_block(out_dict, input_dict, x, weights, stage=2, block='a', strides=1, quant=quant)
    x1 = identity_block(out_dict, input_dict, x, weights, stage=2, block='b', quant=quant)
    x = identity_block(out_dict, input_dict, x1, weights, stage=2, block='c', quant=quant)

    x = conv_block(out_dict, input_dict, x, weights, stage=3, block='a', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=3, block='b', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=3, block='c', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=3, block='d', quant=quant)

    x = conv_block(out_dict, input_dict, x, weights, stage=4, block='a', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=4, block='b', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=4, block='c', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=4, block='d', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=4, block='e', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=4, block='f', quant=quant)

    x = conv_block(out_dict, input_dict, x, weights, stage=5, block='a', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=5, block='b', quant=quant)
    x = identity_block(out_dict, input_dict, x, weights, stage=5, block='c', quant=quant)

    x = avgpool_2d(x, k=7)

    w, b, s = get_weights_biases_scale(weights, 'fc1000_W:0', 'fc1000_b:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s)
    return x
