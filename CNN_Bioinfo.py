from Bio import SeqIO
import numpy as np
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


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
    matrix = []
    with open(input_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            array = np.zeros(7)
            if row[0] == 'A-E':
                array[0] = 1
            if row[0] == 'I-E':
                array[1] = 1
            if row[0] == 'A-P':
                array[2] = 1
            if row[0] == 'I-P':
                array[3] = 1
            if row[0] == 'A-X':
                array[4] = 1
            if row[0] == 'I-X':
                array[5] = 1
            if row[0] == 'UK':
                array[6] = 1
            matrix.append(array)
    return np.array(matrix)


def multiple_labels_to_array(*input_files):
    matrix = []
    for input_file in input_files:
        with open(input_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                array = np.zeros(7)
                if row[0] == 'A-E':
                    array[0] = 1
                if row[0] == 'I-E':
                    array[1] = 1
                if row[0] == 'A-P':
                    array[2] = 1
                if row[0] == 'I-P':
                    array[3] = 1
                if row[0] == 'A-X':
                    array[4] = 1
                if row[0] == 'I-X':
                    array[5] = 1
                if row[0] == 'UK':
                    array[6] = 1
                matrix.append(array)
    return np.array(matrix)


# Read data as fasta format and store it into a a npz archive
def sequences_to_array(input_file):
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    matrix_arr = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        # print(name, sequence)
        matrix = to_OHE(sequence)
        matrix_arr.append(matrix)
    return matrix_arr


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


# Read data as fasta format and store it into a a npz archive
def store_multiple_data(*input_files, output_file):
    matrix_arr = []
    for input_file in input_files:
        fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
        with open(output_file, "a+") as out_file:
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                # print(name, sequence)
                matrix = to_OHE(sequence)
                matrix_arr.append(matrix)
    np.savez(output_file, *matrix_arr)

# Takes a fasta data input file and a csv labels input file and generates a merged npz dataset
def generate_dataset(data_input_file, labels_input_file, output_file):
    data_X = sequences_to_array(data_input_file)

    data_Y = labels_to_array(labels_input_file)

    data_X = np.array(data_X)

    print(data_X.shape)
    print(data_Y.shape)

    data_list = []
    data_list.append(data_X)
    data_list.append(data_Y)
    np.savez(output_file, *data_list)

# Takes a npz input file containing all data and generates the datasets for all the tasks
def generate_dataset_tasks(input_file, *output_files):
    #### DATASET ####
    data_X=[]
    data_Y=[]
    # data_X=data_dict.values()[0]
    # data_Y=data_dict.values()[1]

    data_dict = np.load(input_file)
    for k,v in data_dict.items():
        if k == "arr_0":
            data_X=v
        if k == "arr_1":
            data_Y=v
    data_X=np.array(data_X)
    data_Y=np.array(data_Y)
    print(data_X.shape)
    print(data_Y.shape)

    data_X_AEAP = []
    data_Y_AEAP = []

    data_X_AEIE = []
    data_Y_AEIE = []

    data_X_APIP = []
    data_Y_APIP = []

    data_X_IEIP = []
    data_Y_IEIP = []

    print(data_X.shape[0])

    #### TASK SUBDIVISION ####
    for i in range(data_X.shape[0]):
        if data_Y[i][0] == 1:               # AE - 0
            data_X_AEAP.append(data_X[i])
            data_Y_AEAP.append(np.array([0]))
            data_X_AEIE.append(data_X[i])
            data_Y_AEIE.append(np.array([0]))
        elif data_Y[i][2] == 1:             # AP - 2
            data_X_AEAP.append(data_X[i])
            data_Y_AEAP.append(np.array([1]))
            data_X_APIP.append(data_X[i])
            data_Y_APIP.append(np.array([0]))
        elif data_Y[i][1] == 1:             # IE - 1
            data_X_AEIE.append(data_X[i])
            data_Y_AEIE.append(np.array([1]))
            data_X_IEIP.append(data_X[i])
            data_Y_IEIP.append(np.array([0]))
        elif data_Y[i][3] == 1:             # IP - 3
            data_X_APIP.append(data_X[i])
            data_Y_APIP.append(np.array([1]))
            data_X_IEIP.append(data_X[i])
            data_Y_IEIP.append(np.array([1]))

    data_X_AEAP = np.array(data_X_AEAP)
    data_Y_AEAP = np.array(data_Y_AEAP)

    data_X_AEIE = np.array(data_X_AEIE)
    data_Y_AEIE = np.array(data_Y_AEIE)

    data_X_APIP = np.array(data_X_APIP)
    data_Y_APIP = np.array(data_Y_APIP)

    data_X_IEIP = np.array(data_X_IEIP)
    data_Y_IEIP = np.array(data_Y_IEIP)

    print(data_X_AEAP.shape)
    print(data_Y_AEAP.shape)

    data_list = []
    data_list.append(data_X_AEAP)
    data_list.append(data_Y_AEAP)
    output_file = output_files[0]           #AEAP
    np.savez(output_file, *data_list)

    print(data_X_AEIE.shape)
    print(data_Y_AEIE.shape)

    data_list = []
    data_list.append(data_X_AEIE)
    data_list.append(data_Y_AEIE)
    output_file = output_files[1]           #AEIE
    np.savez(output_file, *data_list)


    print(data_X_APIP.shape)
    print(data_Y_APIP.shape)

    data_list = []
    data_list.append(data_X_APIP)
    data_list.append(data_Y_APIP)
    output_file = output_files[2]           #APIP
    np.savez(output_file, *data_list)


    print(data_X_IEIP.shape)
    print(data_Y_IEIP.shape)

    data_list = []
    data_list.append(data_X_IEIP)
    data_list.append(data_Y_IEIP)
    output_file = output_files[3]           #IEIP
    np.savez(output_file, *data_list)


# Convolution layer + activation
def conv2d(x, W, b, strides=1):
    print("Input: " + str(x.shape))
    print("Weights: " + str(W.shape))
    # Conv2D wrapper, with bias and relu activation
    # The first 1 in strides refers to the image index and the last one refers to the image channel (in this case
    # they both need to be set to 1). SAME as padding makes sure that the kernel can process each pixel, even those
    # in the borders, by adding the needed zero-padding.
    x = tf.nn.conv2d(x, W, strides=[1, strides, 1, 1], padding='SAME')
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

    # First Convolution Layer # 320 - 480 - 960
    w = tf.get_variable('WC1', shape=(8, 1, 4, 32), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC1', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc1"] = w
    biases["bc1"] = b
    convo1 = conv2d(x, w, b, strides=1)
    conv1 = maxpoolseq(convo1)

    # Second Convolution Layer
    w = tf.get_variable('WC2', shape=(8, 1, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC2', shape=(64), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc2"] = w
    biases["bc2"] = b
    convo2 = conv2d(conv1, w, b, strides=1)
    conv2 = maxpoolseq(convo2)

    # Third Convolution Layer
    w = tf.get_variable('WC3', shape=(8, 1, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC3', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
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
    hid_1 = tf.reshape(conv3, [-1, weights["wfcin"].get_shape().as_list()[0]])
    hid_1 = tf.add(tf.matmul(hid_1, w), b)
    hid_1 = tf.nn.relu(hid_1)

    # First Fully Connected Hidden Layer
    # Weights Layer
    w = tf.get_variable("WFCH1", shape=(dim_ch, dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCH1", shape=(dim_ch), initializer=tf.contrib.layers.xavier_initializer())
    weights["wfch1"] = w
    biases["bfch1"] = b
    hid_2 = tf.add(tf.matmul(hid_1, w), b)
    hid_2 = tf.nn.relu(hid_2)

    # Fully Connected Output Layer
    # Weights Layer
    w = tf.get_variable("WFCOUT", shape=(dim_ch, 1), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCOUT", shape=(1), initializer=tf.contrib.layers.xavier_initializer())
    weights["wfcout"] = w
    biases["bfcout"] = b

    # Fully Connected Output Layer: Class Prediction
    out = tf.add(tf.matmul(hid_2, w), b)
    return out


# # input_file = 'data\\bioinfo\\GM12878.fa'
# # output_file = 'data\\bioinfo\\GM12878_in.npz'
#
# input_files = ["data\\bioinfo\\GM12878.fa", "data\\bioinfo\\HelaS3.fa",
#                "data\\bioinfo\\HepG2.fa", "data\\bioinfo\\K562.fa"]
#
# output_file = 'data\\bioinfo\\data_all.npz'
#
# store_multiple_data(*input_files, output_file=output_file)
#
# # input_file = 'data\\bioinfo\\GM12878.csv'
# # output_file = 'data\\bioinfo\\GM12878_in.npy'
#
# input_files = ["data\\bioinfo\\GM12878.csv", "data\\bioinfo\\HelaS3.csv",
#                "data\\bioinfo\\HepG2.csv", "data\\bioinfo\\K562.csv"]
#
# data_Y = multiple_labels_to_array(*input_files)
#
# input_file = "data\\bioinfo\\data_all.npz"
#
# matrices = np.load(input_file)
#
# data_X = []
#
# for m in matrices.items():
#     data_X.append(m[1])
#
# data_X = np.array(data_X)
#
# print(data_X.shape)
# print(data_Y.shape)
#
# data_list = []
# data_list.append(data_X)
# data_list.append(data_Y)
# output_file = 'data\\bioinfo\\dataset.npz'
# np.savez(output_file, *data_list)

# ###### GENERATE DATASETS ######
#
# data_input_file = 'data\\bioinfo\\HelaS3.fa'
#
# labels_input_file = 'data\\bioinfo\\HelaS3.csv'
#
# output_file = 'data\\bioinfo\\HelaS3_data.npz'
#
# generate_dataset(data_input_file, labels_input_file, output_file)
#
# output_files = ['data\\bioinfo\\HelaS3_AEAP.npz', 'data\\bioinfo\\HelaS3_AEIE.npz',
#                 'data\\bioinfo\\HelaS3_APIP.npz', 'data\\bioinfo\\HelaS3_IEIP.npz']
#
# generate_dataset_tasks(output_file, *output_files)



input_file = 'data\\bioinfo\\GM12878_AEAP.npz'
data_dict = np.load(input_file)
for k,v in data_dict.items():
    if k == "arr_0":
        data_X=v
    if k == "arr_1":
        data_Y=v
data_X=np.array(data_X)
data_Y=np.array(data_Y)
print(data_X.shape)
print(data_Y.shape)

train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.3, random_state=7)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

#### HYPER-PARAMETERS ####
training_iters = 200
learning_rate = 0.001
batch_size = 128


#### NETWORK PARAMETERS ####
n_classes = 1   # Number of classes to predict (output_number)


#### DEFINE PLACEHOLDERS ####
# Both placeholders are of type float and the argument filled with None refers to the batch size
x = tf.placeholder("float", [None, 200, 1, 4])
y = tf.placeholder("float", [None, n_classes])


# DEFINE THE CNN MODEL, THE COST FUNCTION AND THE OPTIMIZER
pred = conv_net(x)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# MODEL EVALUATION FUNCTIONS
# # Check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# # and both will be a column vector.
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# # Calculate accuracy across all the given images and average them out.
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(pred)), y), tf.float32))

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
            # print(sess.run(tf.equal(tf.round(tf.nn.sigmoid(pred)), y), feed_dict={x: batch_x, y: batch_y}))
        print("Iter " + str(i) + ":\n" + "Training Error: " + "{:.6f}".format(loss) + ", Training Accuracy: " + "{:.5f}".format(acc))
        #print("Optimization Finished!")


        # Calculate accuracy and loss for the test set (for all 10000 mnist test images)
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_Y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Test Error: " + "{:.6f}".format(valid_loss) + ", Test Accuracy: " + "{:.5f}".format(test_acc) + "\n")
    summary_writer.close()
