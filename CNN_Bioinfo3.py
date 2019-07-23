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


##### DATA PREPROCESSING #####
input_file = "data\\bioinfo\\tasks\\GM12878_AEAPR.npz"
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
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(11, 5)
axs[0].set_title('ROC Curve')
axs[0].set_xlabel("FPR")
axs[0].set_ylabel("TPR")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].set_title('PR Curve')
axs[0].set_xlim([0,1])
axs[0].set_ylim([0,1])
axs[1].set_xlim([0,1])
axs[1].set_ylim([0,1])



##### Keras Implementation ######

#create model Keras
model = Sequential()#add model layers
model.add(Conv2D(32, kernel_size=(8,1), activation='relu', input_shape=(200,1,4)))
model.add(MaxPool2D(pool_size=(2,1), strides=1))
model.add(Conv2D(64, kernel_size=(8,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,1), strides=1))
model.add(Conv2D(128, kernel_size=(8,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,1), strides=1))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1, activation="sigmoid"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=20)

#test the model
preds = model.predict(test_X)

# print(preds)
# print(test_Y)

fpr, tpr, trsh1 = roc_curve(test_Y, preds)

# print(fpr)
# print(tpr)
# print(trsh1)


auroc = auc(fpr, tpr)

p, r, trsh2 = precision_recall_curve(test_Y, preds)

auprc = auc(r, p)

# ROC Curve
axs[0].plot(fpr, tpr, label="auroc")

# PR Curve
axs[1].plot(r, p, label="auprc")

output_file = "GM12878_AEAPR_curves"

plt.savefig(output_file)