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
            data_Y_AEAP.append(np.array([1, 0]))
            data_X_AEIE.append(data_X[i])
            data_Y_AEIE.append(np.array([1, 0]))
        elif data_Y[i][2] == 1:             # AP - 2
            data_X_AEAP.append(data_X[i])
            data_Y_AEAP.append(np.array([0, 1]))
            data_X_APIP.append(data_X[i])
            data_Y_APIP.append(np.array([1, 0]))
        elif data_Y[i][1] == 1:             # IE - 1
            data_X_AEIE.append(data_X[i])
            data_Y_AEIE.append(np.array([0, 1]))
            data_X_IEIP.append(data_X[i])
            data_Y_IEIP.append(np.array([1, 0]))
        elif data_Y[i][3] == 1:             # IP - 3
            data_X_APIP.append(data_X[i])
            data_Y_APIP.append(np.array([0, 1]))
            data_X_IEIP.append(data_X[i])
            data_Y_IEIP.append(np.array([0, 1]))

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

# ###### GENERATE DATASETS ######
#
# output_file = 'data\\bioinfo\\GM12878_data.npz'
#
# output_files = ['data\\bioinfo\\tasksOHE\\GM12878_AEAP_OHE.npz', 'data\\bioinfo\\tasksOHE\\GM12878_AEIE_OHE.npz',
#                 'data\\bioinfo\\tasksOHE\\GM12878_APIP_OHE.npz', 'data\\bioinfo\\tasksOHE\\GM12878_IEIP_OHE.npz']
#
# generate_dataset_tasks(output_file, *output_files)

##### DATA PREPROCESSING #####
input_file = 'data\\bioinfo\\tasksOHE\\GM12878_AEAP_OHE.npz'
data_dict = np.load(input_file)
for k, v in data_dict.items():
    if k == "arr_0":
        data_X = v
    if k == "arr_1":
        data_Y = v
data_X = np.array(data_X)
data_Y = np.array(data_Y)
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


##### Keras Implementation ######

#create model Keras
model = Sequential()#add model layers
model.add(Conv2D(128, kernel_size=(8,1), input_shape=(200,1,4), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(8,1), input_shape=(193,1,128), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,1), strides=1, padding='valid'))
model.add(Conv2D(64, kernel_size=(3,1), input_shape=(96,1,128), padding='valid'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,1), input_shape=(94,1,64), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,1), strides=1, padding='valid'))   # 47
model.add(Flatten())    # 47 x 1 x 64
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(2, activation='softmax'))

opt = Adam(lr=0.001)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)

#compile model using accuracy to measure model performance
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=30, callbacks=[early_stopping])

#test the model
preds = model.predict(test_X)

print(preds)
print(test_Y)

fpr, tpr, trsh1 = roc_curve(test_Y, preds)

auroc = auc(fpr, tpr)

p, r, trsh2 = precision_recall_curve(test_Y, preds)

auprc = auc(r, p)

# ROC Curve
axs[0].plot(fpr, tpr, label="auroc")

# PR Curve
axs[1].plot(r, p, label="auprc")

output_file = "GM12878_AEAP_OHE_curves"

plt.savefig(output_file)
