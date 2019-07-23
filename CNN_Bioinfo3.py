import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def cnn_model(x_w, x_h, x_d, conv_num, hid_num, hid_n_num, ker_r, ker_num, activation, out_num=1, k=2, ker_c=1, maxpooling=True, strides=1):
    model = Sequential()  # add model layers

    ##### CONVOLUTIONAL LAYERS #####
    for i in range(conv_num):
        print("\nConvolution Layer: " + str(i))
        model.add(Conv2D(ker_num, kernel_size=(ker_r, ker_c), activation=activation, input_shape=(x_w, x_h, x_d)))
        if (maxpooling):
            model.add(MaxPool2D(pool_size=(k, 1), strides=strides))
        ker_num *= 2

    ##### FULLY CONNECTED INPUT #####
    model.add(Flatten())

    ##### FULLY CONNECTED HIDDEN LAYERS #####
    for j in range(hid_num):
        model.add(Dense(hid_n_num))

    ##### FULLY CONNECTED OUTPUT #####
    model.add(Dense(out_num, activation="sigmoid"))

    return model


##### DATA PREPROCESSING #####
input_file = "data\\bioinfo\\tasks\\GM12878_AEAP.npz"
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

if input_file.__contains__("AEIE"):
    print("AEIE")
    data_X_temp = []
    data_y_temp = []
    count_zero = 0
    count_one = 0
    for i in range(len(data_Y)):
        if data_Y[i] == 0:
            count_zero += 1
            data_X_temp.append(data_X[i])
            data_y_temp.append(data_Y[i])
        else:
            count_one+=1
            if count_one <= 10000:
                data_X_temp.append(data_X[i])
                data_y_temp.append(data_Y[i])
    print(count_zero, count_one)
    data_X = data_X_temp
    data_Y = data_y_temp
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    print(data_X.shape, data_Y.shape)

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
##### Create figure and subplots #####
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(11, 10)
axs[0][0].set_title('ROC Curve - Relu')
axs[0][0].set_xlabel("FPR")
axs[0][0].set_ylabel("TPR")
axs[0][1].set_xlabel("Recall")
axs[0][1].set_ylabel("Precision")
axs[0][1].set_title('PR Curve - Relu')
axs[1][0].set_title('ROC Curve - Tanh')
axs[1][0].set_xlabel("FPR")
axs[1][0].set_ylabel("TPR")
axs[1][1].set_xlabel("Recall")
axs[1][1].set_ylabel("Precision")
axs[1][1].set_title('PR Curve - Tanh')
axs[0][0].set_xlim([-0.1, 1.1])
axs[0][0].set_ylim([-0.1, 1.1])
axs[0][1].set_xlim([-0.1, 1.1])
axs[0][1].set_ylim([-0.1, 1.1])
axs[1][0].set_xlim([-0.1, 1.1])
axs[1][0].set_ylim([-0.1, 1.1])
axs[1][1].set_xlim([-0.1, 1.1])
axs[1][1].set_ylim([-0.1, 1.1])
plt.subplots_adjust(wspace=0.3)


##### Keras Implementation ######

# #create model Keras
# model = Sequential()#add model layers
# model.add(Conv2D(32, kernel_size=(8,1), activation='relu', input_shape=(200, 1 ,4)))
# model.add(MaxPool2D(pool_size=(2,1), strides=1))
# model.add(Conv2D(64, kernel_size=(8,1), activation='relu'))
# model.add(MaxPool2D(pool_size=(2,1), strides=1))
# model.add(Conv2D(128, kernel_size=(8,1), activation='relu'))
# model.add(MaxPool2D(pool_size=(2,1), strides=1))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(1, activation="sigmoid"))

x_w = train_X.shape[1]
x_h = train_X.shape[2]
x_d = train_X.shape[3]

act_functions = ["relu", "tanh"]
# maxpooling = [True, False]
conv_layers = [2, 3]
hid_layers = [1, 2]
hid_n_num = [32, 64]
ker_r = 8
ker_num = 32

for a in range(len(act_functions)):
    for c_l in conv_layers:
        for h_l in hid_layers:
            for h_n_n in hid_n_num:

                model = cnn_model(x_w, x_h, x_d, conv_num=c_l, hid_num=h_l, hid_n_num=hid_n_num, ker_r=ker_r,
                                  ker_num=ker_num, activation=act_functions[a])

                #compile model using accuracy to measure model performance
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                #train the model
                model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=20)

                #test the model
                preds = model.predict(test_X)

                fpr, tpr, trsh1 = roc_curve(test_Y, preds)

                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.insert(tpr, 0, 0.0)
                fpr = np.append(fpr, 1.0)
                tpr = np.append(tpr, 1.0)

                auroc = auc(fpr, tpr)

                p, r, trsh2 = precision_recall_curve(test_Y, preds)

                r = np.insert(r, 0, 1.0)
                p = np.insert(p, 0, 0.0)
                r = np.append(r, 0.0)
                p = np.append(p, 1.0)

                auprc = auc(r, p)

                # ROC Curve
                axs[a][0].plot(fpr, tpr, label="auroc")

                # PR Curve
                axs[a][1].plot(r, p, label="auprc")

    axs[0][0].legend(loc="lower right", fontsize='x-small')
    axs[0][1].legend(loc="lower left", fontsize='x-small')
    axs[1][0].legend(loc="lower right", fontsize='x-small')
    axs[1][1].legend(loc="lower left", fontsize='x-small')

output_file = "GM12878_AEAP_curves"

plt.savefig(output_file)