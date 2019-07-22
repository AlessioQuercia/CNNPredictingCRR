from Bio import SeqIO
import numpy as np
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt_auroc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def compute_TPR(TP, FN):
    return TP/(TP + FN)


def compute_FPR(FP, TN):
    return FP/(FP + TN)


def compute_precision(TP, FP):
    return TP/(TP+FP)


def compute_recall(TP, FN):
    return TP/(TP + FN)


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
def conv2d(x, W, b, activation, strides=1):
    print("Input: " + str(x.shape))
    print("Weights: " + str(W.shape))
    # Conv2D wrapper, with bias and relu activation
    # The first 1 in strides refers to the image index and the last one refers to the image channel (in this case
    # they both need to be set to 1). SAME as padding makes sure that the kernel can process each pixel, even those
    # in the borders, by adding the needed zero-padding.
    x = tf.nn.conv2d(x, W, strides=[1, strides, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    if activation == "tanh":
        out = tf.nn.tanh(x)
    elif activation == "sigmoid":
        out = tf.nn.sigmoid(x)
    else:
        out = tf.nn.relu(x)
    return out


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
    convo1 = conv2d(x, w, b, "relu", strides=1)
    conv1 = maxpoolseq(convo1)

    # Second Convolution Layer
    w = tf.get_variable('WC2', shape=(8, 1, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC2', shape=(64), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc2"] = w
    biases["bc2"] = b
    convo2 = conv2d(conv1, w, b, "relu", strides=1)
    conv2 = maxpoolseq(convo2)

    # Third Convolution Layer
    w = tf.get_variable('WC3', shape=(8, 1, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable('BC3', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
    weights["wc3"] = w
    biases["bc3"] = b
    convo3 = conv2d(conv2, w, b, "relu", strides=1)
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


# Convolutional Neural Network model
def conv_net_gen(x, conv_num, hid_num, hid_n_num, ker_r, ker_c, ker_ch, ker_num, out_num, k, activation, count, maxpooling=True, strides=1):
    inp = x
    weights = {}
    biases = {}
    for i in range(conv_num):
        print("\nConvolution Layer: " + str(i))
        # Weights Layer
        w = tf.get_variable('WC' + str(i+count), shape=(ker_r, ker_c, ker_ch, ker_num), initializer=tf.contrib.layers.xavier_initializer())
        # Bias Layer
        b = tf.get_variable('BC' + str(i+count), shape=(ker_num), initializer=tf.contrib.layers.xavier_initializer())

        weights["wc"+str(i)] = w
        biases["bc"+str(i)] = b

        # Convolution Layer
        # Convolution: we pass the input inp, the weights w and biases b.
        conv = conv2d(inp, w, b, activation, strides)
        if (maxpooling):
            conv = maxpoolseq(conv, k=k)

        inp = conv
        ker_ch = ker_num
        ker_num = ker_num*2

    dim_r = inp.shape[1]
    dim_c = inp.shape[2]
    dim_ch = inp.shape[3]

    # Weights Layer
    w = tf.get_variable("WFCIN" + str(count) , shape=(dim_r * dim_c * dim_ch, hid_n_num), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCIN" + str(count), shape=(hid_n_num), initializer=tf.contrib.layers.xavier_initializer())

    weights["wfcin"] = w
    biases["bfcin"] = b

    # Reshape into 1D
    fcl = tf.reshape(inp, [-1, weights["wfcin"].get_shape().as_list()[0]])
    # Fully Connected Input Layer
    print("\nFully Connected Input Layer:")
    print("Input: " + str(inp.shape))
    print("Weights: " + str(w.shape))
    fcl = tf.add(tf.matmul(fcl, w), b)
    if activation == "tanh":
        fcl = tf.nn.tanh(fcl)
    elif activation == "sigmoid":
        fcl = tf.nn.sigmoid(fcl)
    else:
        fcl = tf.nn.relu(fcl)

    # Fully Connected Hidden Layers
    for j in range(hid_num):
        # Weights Layer
        w = tf.get_variable("WFCH" + str(j + count), shape=(hid_n_num, hid_n_num), initializer=tf.contrib.layers.xavier_initializer())
        # Bias Layer
        b = tf.get_variable("BFCH" + str(j + count), shape=(hid_n_num), initializer=tf.contrib.layers.xavier_initializer())

        weights["wfch" + str(j)] = w
        biases["bfch" + str(j)] = b

        print("\nFully Connected Hidden Layer: " + str(j))
        print("Input: " + str(fcl.shape))
        print("Weights: " + str(w.shape))

        fcl = tf.add(tf.matmul(fcl, w), b)
        if activation == "tanh":
            fcl = tf.nn.tanh(fcl)
        elif activation == "sigmoid":
            fcl = tf.nn.sigmoid(fcl)
        else:
            fcl = tf.nn.relu(fcl)

    # Weights Layer
    w = tf.get_variable("WFCOUT" + str(count), shape=(hid_n_num, out_num), initializer=tf.contrib.layers.xavier_initializer())
    # Bias Layer
    b = tf.get_variable("BFCOUT" + str(count), shape=(out_num), initializer=tf.contrib.layers.xavier_initializer())

    weights["wfcout"] = w
    biases["bfcout"] = b

    print("\nFully Connected Output Layer:")
    print("Input: " + str(fcl.shape))
    print("Weights: " + str(w.shape))

    # Fully Connected Output Layer: Class Prediction
    out = tf.add(tf.matmul(fcl, w), b)

    return out


def depict_ROC_curve(label, color, filename, fpr, tpr, auc, xlabel, ylabel, title, linestyle, linewidth, randomline=False, save=False):
    """
    :type color: string (hex color code)
    :type fname: string
    :type randomline: boolean
    """

    plt.figure(figsize=(4, 4), dpi=80)

    setup_ROC_curve_plot(plt, xlabel, ylabel, title)
    add_ROC_curve(plt, color, label, tpr, fpr, auc)
    if (save):
        save_ROC_curve_plot(plt, filename, randomline)


def setup_ROC_curve_plot(plt, xlabel, ylabel, title):
    """
    :type plt: matplotlib.pyplot
    """

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)


def add_ROC_curve(plt, color, linestyle, linewidth, label, tpr, fpr, auc):
    """
    :type plt: matplotlib.pyplot
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :type color: string (hex color code)
    :type label: string
    """

    roc_label = '({} ={:.3f})'.format(label, auc)
    plt.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth, color=color, label=roc_label)


def save_ROC_curve_plot(plt, filename, randomline=True):
    """
    :type plt: matplotlib.pyplot
    :type fname: string
    :type randomline: boolean
    """

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(filename)


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


##### DATA PREPROCESSING #####
input_file = 'data\\bioinfo\\tasks\\GM12878_AEIE.npz'
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

##### Create figure and subplots #####
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(11, 5)
axs[0].set_title('ROC Curve')
axs[0].set_xlabel("FPR")
axs[0].set_ylabel("TPR")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].set_title('PR Curve')
plt.subplots_adjust(wspace=0.3)

##### Create parameters vectors #####

act_functions = ["relu", "sigmoid", "tanh"]
conv_layers = [2, 3]
hid_layers = [1, 2]
hid_n_num = [64]

ker_r = 8
ker_c = 1
ker_ch = 4
ker_num = 32
k = 2

count = 0

for a_f in act_functions:
    for c_l in conv_layers:
        for h_l in hid_layers:
            for h_n_n in hid_n_num:

                #### HYPER-PARAMETERS ####
                training_iters = 10
                learning_rate = 0.001
                batch_size = 100


                #### NETWORK PARAMETERS ####
                n_classes = 1   # Number of classes to predict (output_number)


                #### DEFINE PLACEHOLDERS ####
                # Both placeholders are of type float and the argument filled with None refers to the batch size
                x = tf.placeholder("float", [None, 200, 1, 4])
                y = tf.placeholder("float", [None, n_classes])


                # DEFINE THE CNN MODEL, THE COST FUNCTION AND THE OPTIMIZER
                pred = conv_net_gen(x, c_l, h_l-1, h_n_n, ker_r, ker_c, ker_ch, ker_num, n_classes, k, a_f, count)
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


                # MODEL EVALUATION FUNCTIONS
                # # Check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
                # # and both will be a column vector.
                # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # # Calculate accuracy across all the given images and average them out.
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(pred)), y), tf.float32))
                _, auroc_mes = tf.metrics.auc(y, tf.round(tf.sigmoid(pred)), curve='ROC', name="AUROC")
                _, auprc_mes = tf.metrics.auc(y, tf.round(tf.sigmoid(pred)), curve='PR', name="AUPRC")
                _, tp_mes = tf.metrics.true_positives(y, tf.round(tf.sigmoid(pred)))
                _, tn_mes = tf.metrics.true_negatives(y, tf.round(tf.sigmoid(pred)))
                _, fp_mes = tf.metrics.false_positives(y, tf.round(tf.sigmoid(pred)))
                _, fn_mes = tf.metrics.false_negatives(y, tf.round(tf.sigmoid(pred)))

                # INITIALIZING THE VARIABLES
                init = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                tvars = tf.local_variables()


                # AUROC
                tpr_list = []
                fpr_list = []
                auroc_list = []

                # AUPRC
                p_list = []
                r_list = []
                auprc_list = []

                # TRAINING AND TESTING THE MODEL
                with tf.Session() as sess:
                    sess.run(init)
                    sess.run(init_local)
                    train_loss = []
                    test_loss = []
                    train_accuracy = []
                    test_accuracy = []
                    train_auroc = []
                    test_auroc = []
                    train_auprc = []
                    test_auprc = []
                    y_preds = []
                    y_trues = []
                    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
                    for i in range(training_iters):
                        # sess.run(tf.local_variables_initializer())
                        for batch in range(len(train_X)//batch_size):
                            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
                            # Run optimization op (backprop).
                                # Calculate batch loss and accuracy
                            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                            loss, acc, auroc, auprc = sess.run([cost, accuracy, auroc_mes, auprc_mes], feed_dict={x: batch_x, y: batch_y})
                            # print(sess.run(tf.equal(tf.round(tf.nn.sigmoid(pred)), y), feed_dict={x: batch_x, y: batch_y}))
                        print("Iter " + str(i) + ":\n" + "Training Error: " + "{:.6f}".format(loss) + ", Training Accuracy: " + "{:.6f}".format(acc))

                        #print("Optimization Finished!")

                        sess.run(tf.local_variables_initializer())

                        y_pred, y_true, tp, tn, fp, fn, test_acc,test_loss, test_auroc, test_auprc = \
                            sess.run([tf.round(tf.nn.sigmoid(pred)), y, tp_mes, tn_mes, fp_mes, fn_mes, accuracy, cost, auroc_mes, auprc_mes],
                                     feed_dict={x: test_X,y : test_Y})

                        y_preds.append(y_pred)
                        y_trues.append(y_true)

                        print(test_X.shape, tp, tn, fp, fn, compute_TPR(tp, fn), compute_FPR(fp, tn), compute_precision(tp, fp), compute_recall(tp, fn))

                        print("Test Error: " + "{:.6f}".format(test_loss) + ", Test Accuracy: " + "{:.6f}".format(test_acc)
                              + ", Test AUROC: " + "{:.6f}".format(test_auroc) + ", Test AUPRC: " + "{:.6f}".format(test_auprc) + "\n")

                        # print(sess.run([tf.round(tf.nn.sigmoid(pred)), y], feed_dict={x: batch_x, y: batch_y}))
                        # print(roc_curve(sess.run(y, feed_dict={x: test_X,y : test_Y}), sess.run(tf.round(tf.nn.sigmoid(pred)))))

                        # tpr_list.append(compute_TPR(tp, fn))
                        # fpr_list.append(compute_FPR(fp, tn))
                        # auroc_list.append(test_auroc)
                        # p_list.append(compute_precision(tp, fp))
                        # r_list.append(compute_recall(tp, fn))
                        # auprc_list.append(test_auprc)

                        # train_loss.append(loss)
                        # test_loss.append(test_loss)
                        # train_accuracy.append(acc)
                        # test_accuracy.append(test_acc)
                        # train_auroc.append(auroc)
                        # test_auroc.append(test_auroc)
                        # train_auprc.append(auprc)
                        # test_auprc.append(test_auprc)
                        count += 1
                    summary_writer.close()

                # print(auc(fpr_list, tpr_list))
                fpr_list = []
                tpr_list = []
                fpr_list.append(0.0)
                tpr_list.append(0.0)

                for i in range(len(y_preds)):
                    # print(roc_curve(y_trues[i], y_preds[i]))
                    fpr_list.append(roc_curve(y_trues[i], y_preds[i])[0][1])
                    tpr_list.append(roc_curve(y_trues[i], y_preds[i])[1][1])

                tpr_list.append(1.0)
                fpr_list.append(1.0)

                fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))

                fpr_list, tpr_list = (list(t) for t in zip(*sorted(zip(fpr_list, tpr_list))))

                # print(fpr_list)
                # print(tpr_list)

                auroc = auc(fpr_list, tpr_list)

                r_list = []
                p_list = []
                r_list.append(0.0)
                p_list.append(1.0)

                for i in range(len(y_preds)):
                    # print(precision_recall_curve(y_trues[i], y_preds[i]))
                    r_list.append(precision_recall_curve(y_trues[i], y_preds[i])[1][1])
                    p_list.append(precision_recall_curve(y_trues[i], y_preds[i])[0][1])

                r_list, p_list = zip(*sorted(zip(r_list, p_list)))

                r_list, p_list = (list(t) for t in zip(*sorted(zip(r_list, p_list))))

                r_list.append(1.0)
                p_list.append(0.0)

                # print(r_list)
                # print(p_list)

                auprc = auc(r_list, p_list)

                roc_label = a_f + " " + str(c_l) + "CL" + " " + str(h_l) + "HL" + " " + str(h_n_n) + "N" + " " + "AUROC: " + str(auroc)
                pr_label = a_f + " " + str(c_l) + "CL" + " " + str(h_l) + "HL" + " " + str(h_n_n) + "N" + " " + "AUPRC: " + str(auprc)

                # ROC Curve
                axs[0].plot(fpr_list, tpr_list, label=roc_label)

                # PR Curve
                axs[1].plot(r_list, p_list, label=pr_label)


axs[0].legend(loc="lower right")
axs[1].legend(loc="lower right")

# auroc_output_file = "data\\bioinfo\\tasks\\results\\GM12878_AEIE_auroc"
#
# auprc_output_file = "data\\bioinfo\\tasks\\results\\GM12878_AEIE_auprc"

plt.savefig("data\\bioinfo\\tasks\\results\\GM12878_AEIE_curves")

    # depict_ROC_curve("AUROC", 'blue', auroc_output_file, fpr_list, tpr_list, auroc, "FPR", "TPR", "ROC Curve", "dashed", 1)
    #
    # depict_ROC_curve("AUPRC", 'blue', auprc_output_file, r_list, p_list, auprc, "Recall", "Precision", "PRC Curve", "dashed", 1)



###### Keras Implementation ######

# #create model Keras
# model = Sequential()#add model layers
# model.add(Conv2D(32, kernel_size=(8,1), activation='relu', input_shape=(200,1,4)))
# model.add(Conv2D(64, kernel_size=(8,1), activation='relu'))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Dense(1, activation='sigmoid'))
#
# #compile model using accuracy to measure model performance
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# #train the model
# model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=20)
#
# #test the model
# model.predict(test_X)


