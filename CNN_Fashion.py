import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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


# Read the data
data = input_data.read_data_sets('data/fashion',one_hot=True)


# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))


# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Show both the images
#plt.show()

print(data.train.images[0])

print("Max value: {value}".format(value=max(data.train.images[0])))

print("Min value: {value}".format(value=min(data.train.images[0])))

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)

print("Train_X shape: {shape}".format(shape=train_X.shape))
print("Test_X shape: {shape}".format(shape=test_X.shape))

# print(train_X.shape, test_X.shape)

train_Y = data.train.labels
test_Y = data.test.labels

print("Train_Y shape: {shape}".format(shape=train_Y.shape))
print("Test_Y shape: {shape}".format(shape=test_Y.shape))

# print(train_X.shape, test_X.shape)


# HYPER-PARAMETERS
training_iters = 200
learning_rate = 0.001
batch_size = 128

# NETWORK PARAMETERS
n_input = 28    # MNIST data input (img shape: 28*28)

n_classes = 10  # MNIST total classes (0-9 digits)

# DEFINE PLACEHOLDERS
#both placeholders are of type float and the argument filled with None refers to the batch size
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])

# DEFINE WEIGHTS AND BIASES

# # 3 convolution layers (including convolution and pooling) + 2 fully connected layers
# weights = {
#     'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),         # first convolution layer
#     'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),        # second convolution layer
#     'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),       # third convolution layer
#     'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),      # first fully connected layer
#     'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),    # output layer
# }
# # 5 bias layers
# biases = {
#     'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),               # first bias layer
#     'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),               # second bias layer
#     'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),              # third bias layer
#     'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),              # fourth bias layer
#     'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),               # fifth bias layer
# }


# DEFINE THE CNN MODEL, THE COST FUNCTION AND THE OPTIMIZER

pred = conv_net_gen(x, 3, 2, 3, 3, 1, 32, 10, 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# MODEL EVALUATION FUNCTIONS

# Check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# INITIALIZING THE VARIABLES
init = tf.global_variables_initializer()


# TRAINING AND TESTING THE MODEL

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ":\n" + "Training Error: " + "{:.6f}".format(loss) + ", Training Accuracy: " + "{:.5f}".format(acc))
        #print("Optimization Finished!")


        # Calculate accuracy and loss for the test set (for all 10000 mnist test images)
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_Y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Test Error: " + "{:.6f}".format(valid_loss) + ", Training Accuracy: " + "{:.5f}".format(test_acc) + "\n")
    summary_writer.close()
