from Bio import SeqIO
import numpy as np
import tensorflow as tf
import csv


# Convert a sequence to its One Hot Encoding representation
def to_OHE(sequence):
    matrix = np.zeros((200, 1, 4))
    # ACGT = 0123
    for c in range(len(sequence)):
        if sequence[c] == 'A' or sequence[c] == 'a':
            matrix[c][0][0] = 1
        elif sequence[c] == 'C' or sequence[c] == 'c':
            matrix[c][0][1] = 1
        elif sequence[c] == 'G' or sequence[c] == 'g':
            matrix[c][0][2] = 1
        elif sequence[c] == 'T' or sequence[c] == 't':
            matrix[c][0][3] = 1
    return matrix


def labels_to_array(input_file):
    array = []
    with open(input_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if row[0] == 'A-E':
                array.append(0)
            if row[0] == 'I-E':
                array.append(1)
            if row[0] == 'A-P':
                array.append(2)
            if row[0] == 'I-P':
                array.append(3)
            if row[0] == 'A-X':
                array.append(4)
            if row[0] == 'I-X':
                array.append(5)
            if row[0] == 'UK':
                array.append(6)
    return array




# Read data as fasta format and store it into a a npz archive
def store_data(input_file, output_file):
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    with open(output_file, "a+") as out_file:
        matrix_arr = []
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            # print(name, sequence)
            matrix = to_OHE(sequence)
            matrix_arr.append(matrix)

        np.savez(output_file, *matrix_arr)


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
def maxpoolseq(x, k=2):
    # Apply max pooling to both width and height if using images ([1, k, k, 1] ), just to the length if using sequences ( [1, k, 1, 1] )
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')


# Convolutional Neural Network model
def conv_net(x):
    weights = {}
    biases = {}

    # First Convolution Layer
    w = tf.get_variable('WC1', shape=(8, 1, 4, 320), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC1', shape=(320), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc1"] = w
    biases["bc1"] = b
    convo1 = conv2d(x, w, b, strides=1)
    conv1 = maxpoolseq(convo1)

    # Second Convolution Layer
    w = tf.get_variable('WC2', shape=(8, 1, 320, 480), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC2', shape=(480), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc2"] = w
    biases["bc2"] = b
    convo2 = conv2d(conv1, w, b, strides=1)
    conv2 = maxpoolseq(convo2)

    # Third Convolution Layer
    w = tf.get_variable('WC2', shape=(8, 1, 480, 960), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC2', shape=(960), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc3"] = w
    biases["bc3"] = b
    convo3 = conv2d(conv2, w, b, strides=1)
    conv3 = maxpoolseq(convo3)


    # First Fully Connected Hidden Layer
    dim_r = conv3.shape[1]
    dim_c = conv3.shape[2]
    dim_ch = conv3.shape[3]
    # Weights Layer
    w = tf.get_variable("WFCIN" , shape=(dim_r * dim_c * dim_ch, dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCIN", shape=(dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    weights["wfcin"] = w
    biases["bfcin"] = b
    # Reshape into 1D
    fcl = tf.reshape(conv3, [-1, weights["wfcin"].get_shape().as_list()[0]])
    fcl = tf.add(tf.matmul(fcl, w), b)
    fcl = tf.nn.relu(fcl)

    # First Fully Connected Hidden Layer
    # Weights Layer
    w = tf.get_variable("WFCH1", shape=(dim_ch, dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCH1", shape=(dim_ch), initializer=tf.contrib.layers.xavier_initializer())

    # Fully Connected Output Layer
    # Weights Layer
    w = tf.get_variable("WFCOUT", shape=(dim_ch, 2), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCOUT", shape=(2), initializer=tf.contrib.layers.xavier_initializer())
    weights["wfcout"] = w
    biases["bfcout"] = b

    # Fully Connected Output Layer: Class Prediction
    out = tf.add(tf.matmul(fcl, w), b)
    return out

input_file = 'data\\bioinfo\\GM12878.csv'

data_Y = labels_to_array(input_file)

input_file = 'data\\bioinfo\\GM12878.fa'
output_file = 'data\\bioinfo\\GM12878_in.npz'

#store_data(input_file, output_file)

matrices = np.load(output_file)

data_X = []

for m in matrices.items():
    data_X.append(m[1])

data_X = np.array(data_X)
data_Y = np.array(data_Y)

print(data_X.shape)
print(data_Y.shape)


#### DATASET ####
# # Reshape training and testing image
# train_X = data.train.images.reshape(-1, 28, 28, 1)
# test_X = data.test.images.reshape(-1, 28, 28, 1)
#
# train_Y = data.train.labels
# test_Y = data.test.labels

# #### HYPER-PARAMETERS ####
# training_iters = 200
# learning_rate = 0.001
# batch_size = 128
#
#
# #### NETWORK PARAMETERS ####
# n_input = 28    # MNIST data input (img shape: 28*28)
# n_classes = 2   # Number of classes to predict (output_number)
# conv_num = 3    # Number of convolution layers
# full_h_num = 2  # Number of hidden layers in the fully connected neural network at the end
# ker_r = 8       # Kernel rows number
# ker_c = 1       # Kernel columns number
# ker_ch = 4      # Kernel channels number
# ker_num = 32    # Kernel initial number
# k = 2           # MaxPool number
#
#
# #### DEFINE PLACEHOLDERS ####
# # Both placeholders are of type float and the argument filled with None refers to the batch size
# x = tf.placeholder("float", [None, 200, 1, 4])
# y = tf.placeholder("float", [None, n_classes])
#
#
# # DEFINE THE CNN MODEL, THE COST FUNCTION AND THE OPTIMIZER
# pred = conv_net(x)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#
# # MODEL EVALUATION FUNCTIONS
# # Check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# # and both will be a column vector.
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# # Calculate accuracy across all the given images and average them out.
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
# # INITIALIZING THE VARIABLES
# init = tf.global_variables_initializer()


# # TRAINING AND TESTING THE MODEL
# with tf.Session() as sess:
#     sess.run(init)
#     train_loss = []
#     test_loss = []
#     train_accuracy = []
#     test_accuracy = []
#     summary_writer = tf.summary.FileWriter('./Output', sess.graph)
#     for i in range(training_iters):
#         for batch in range(len(train_X)//batch_size):
#             batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
#             batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
#             # Run optimization op (backprop).
#                 # Calculate batch loss and accuracy
#             opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#             loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
#         print("Iter " + str(i) + ":\n" + "Training Error: " + "{:.6f}".format(loss) + ", Training Accuracy: " + "{:.5f}".format(acc))
#         #print("Optimization Finished!")
#
#
#         # Calculate accuracy and loss for the test set (for all 10000 mnist test images)
#         test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_Y})
#         train_loss.append(loss)
#         test_loss.append(valid_loss)
#         train_accuracy.append(acc)
#         test_accuracy.append(test_acc)
#         print("Test Error: " + "{:.6f}".format(valid_loss) + ", Training Accuracy: " + "{:.5f}".format(test_acc) + "\n")
#     summary_writer.close()