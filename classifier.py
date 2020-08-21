from buildNet import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

modelOutput = 'assets/model.h5'

INIT_LR = 1e-3
EPOCHS = 10
BS = 128

print(r"Access to MNIST")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

bin = LabelBinarizer()
trainLabels = bin.fit_transform(trainLabels)
testLabels = bin.transform(testLabels)

print(r"Compile model")
optimizer = Adam(lr=INIT_LR)
model = SudokuNet.build_model(28, 28, 1, 10)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(r"Training the network")
net = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=BS, epochs=EPOCHS, verbose=1)

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
plt.savefig('assets/plot.png')
