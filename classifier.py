from buildNet import *
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import backend as K
import tensorflow as tf
import datetime

import numpy as np
import matplotlib.pyplot as plt

import struct
from array import array

num_train = 60000
num_test = 10000
num_classes = 10

data_train = 'infimnist/train-images-idx3-ubyte'
label_train = 'infimnist/train-labels-idx1-ubyte'
data_test = 'infimnist/t10k-images-idx3-ubyte'
label_test = 'infimnist/t10k-labels-idx1-ubyte'

modelOutput = 'assets/large_model.h5'

INIT_LR = 1e-3
EPOCHS = 10
BS = 128


def label_read(path):
    with open(path, "rb") as binary_file:
        y_ = np.array(array("B", binary_file.read()))
    return y_


def image_read(path):
    with open(path, "rb") as binary_file:
        images = []
        emnistRotate = False
        magic, size, rows, cols = struct.unpack(">IIII", binary_file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,''got {}'.format(magic))
        for i in range(size):
            images.append([0] * rows * cols)
        image_data = array("B", binary_file.read())
        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

            # for some reason EMNIST is mirrored and rotated
            if emnistRotate:
                x = image_data[i * rows * cols:(i + 1) * rows * cols]

                subs = []
                for r in range(rows):
                    subs.append(x[(rows - r) * cols - cols:(rows - r) * cols])

                l = list(zip(*reversed(subs)))
                fixed = [item for sublist in l for item in sublist]
                images[i][:] = fixed
    x = []
    for image in images:
        x.append(np.array(image))
    x_ = np.array(x)
    return x_


print(r"Access to MNIST")
trainData = image_read(data_train)
trainLabels = label_read(label_train)
testData = image_read(data_test)
testLabels = label_read(label_test)

from keras.utils import np_utils

trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
testData = testData.reshape(testData.shape[0], 28, 28, 1)

trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

trainLabels = np_utils.to_categorical(trainLabels, 60000)  # One-hot encode the labels
testLabels = np_utils.to_categorical(testLabels, 10000)  # One-hot encode the labels

bin = LabelBinarizer()
trainLabels = bin.fit_transform(trainLabels)
testLabels = bin.transform(testLabels)

print(r"Compile model")
optimizer = Adam(lr=INIT_LR)
model = SudokuNet.build_model(28, 28, 1, 10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print(len(trainData), len(trainLabels), len(testData), len(testLabels))

print(r"Training the network")
net = model.fit(x=trainData, y=trainLabels, validation_data=(testData, testLabels), batch_size=BS, epochs=EPOCHS,
                verbose=1, callbacks=[tensorboard_callback])

print(r"Eval the network")
pred = model.predict(testData)
print(
    classification_report(testLabels.argmax(axis=1), pred.argmax(axis=1), target_names=[str(x) for x in range(0, 10)]))

print(r"Serialization and saving")
model.save(modelOutput, save_format='h5')

plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
N = EPOCHS
plt.plot(np.arange(0, N), net.history["loss"], label=r"Training loss")
plt.plot(np.arange(0, N), net.history["val_loss"], label=r"Validation loss")
plt.plot(np.arange(0, N), net.history["accuracy"], label=r"Training accuracy")
plt.plot(np.arange(0, N), net.history["val_accuracy"], label=r"Validation accuracy")
plt.title(r"Training Loss and Accuracy")
plt.xlabel(r"Epoch #")
plt.ylabel(r"Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig('assets/large_model_plot.png')
