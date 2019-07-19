import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


# Convolution layer + activation
def conv2d(x, W, b, strides=1):
    print("Input: " + str(x.shape))
    print("Weights: " + str(W.shape))
    # Conv2D wrapper, with bias and relu activation
    # The first 1 in strides refers to the image index and the last one refers to the image channel (in this case
    # they both need to be set to 1). SAME as padding makes sure that the kernel can process each pixel, even those
    # in the borders, by adding the needed zero-padding.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# Pooling layer
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


# Convolutional Neural Network model
def conv_net(x, weights, biases):
    # First Convolution
    # First Convolution Layer: we pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # First Max Pooling Layer(down-sampling): this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Second Convolution
    # Second Convolution Layer: we pass the output of the first convolution, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Second Max Pooling Layer(down-sampling): this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    # Third Convolution
    # Third Convolution Layer: we pass the output of the second convolution, weights wc3 and bias bc3.
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Third Max Pooling Layer(down-sampling): this chooses the max value from a 2*2 matrix window and outputs a 4*4 matrix.
    conv3 = maxpool2d(conv3, k=2)


    # First Fully Connected (Hidden) Layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Output Layer: class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Convolutional Neural Network model
def conv_net_gen(x, conv_num, full_h_num, ker_r, ker_c, ker_ch, ker_num, out_num, k, strides=1):
    inp = x
    weights = {}
    biases = {}
    for i in range(conv_num):
        print("\nConvolution Layer: " + str(i))
        # Weights Layer
        w = tf.get_variable('WC' + str(i), shape=(ker_r, ker_c, ker_ch, ker_num), initializer=tf.contrib.layers.xavier_initializer())
        # Bias Layer
        b = tf.get_variable('BC' + str(i), shape=(ker_num), initializer=tf.contrib.layers.xavier_initializer())

        weights["wc"+str(i)] = w
        biases["bc"+str(i)] = b

        # Convolution Layer
        # Convolution: we pass the input inp, the weights w and biases b.
        conv = conv2d(inp, w, b, strides)
        # Max Pooling Layer(down-sampling): this chooses the max value from a k*k matrix window and outputs a n/2*n/2 matrix.
        conv = maxpool2d(conv, k=k)

        inp = conv
        ker_ch = ker_num
        ker_num = ker_num*2

    dim_r = inp.shape[1]
    dim_c = inp.shape[2]
    dim_ch = inp.shape[3]

    # Weights Layer
    w = tf.get_variable("WFCIN" , shape=(dim_r * dim_c * dim_ch, dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCIN", shape=(dim_ch), initializer=tf.contrib.layers.xavier_initializer())

    weights["wfcin"] = w
    biases["bfcin"] = b

    # Reshape into 1D
    fcl = tf.reshape(inp, [-1, weights["wfcin"].get_shape().as_list()[0]])
    # Fully Connected Input Layer
    print("\nFully Connected Input Layer:")
    print("Input: " + str(inp.shape))
    print("Weights: " + str(w.shape))
    fcl = tf.add(tf.matmul(fcl, w), b)
    fcl = tf.nn.relu(fcl)

    # Fully Connected Hidden Layers
    for j in range(full_h_num):
        # Weights Layer
        w = tf.get_variable("WFCH" + str(j), shape=(dim_ch, dim_ch), initializer=tf.contrib.layers.xavier_initializer())
        # Bias Layer
        b = tf.get_variable("BFCH" + str(j), shape=(dim_ch), initializer=tf.contrib.layers.xavier_initializer())

        weights["wfch" + str(j)] = w
        biases["bfch" + str(j)] = b

        print("\nFully Connected Hidden Layer: " + str(j))
        print("Input: " + str(fcl.shape))
        print("Weights: " + str(w.shape))

        fcl = tf.add(tf.matmul(fcl, w), b)
        fcl = tf.nn.relu(fcl)

    # Weights Layer
    w = tf.get_variable("WFCOUT", shape=(dim_ch, out_num), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCOUT", shape=(out_num), initializer=tf.contrib.layers.xavier_initializer())

    weights["wfcout"] = w
    biases["bfcout"] = b

    print("\nFully Connected Output Layer:")
    print("Input: " + str(fcl.shape))
    print("Weights: " + str(w.shape))

    # Fully Connected Output Layer: Class Prediction
    out = tf.add(tf.matmul(fcl, w), b)

    return out


data_X = ["mela", "pera", "banana", "pesca", "ananas", "albicocca"]
random.Random(7).shuffle(data_X)
print(data_X)

data_Y = [1, 2, 3, 4, 5, 6]
random.Random(7).shuffle(data_Y)
print(data_Y)