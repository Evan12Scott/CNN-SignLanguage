import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models

#Read data file into pandas dataframe
tr_data = pd.read_csv("./Data/sign_mnist_train/sign_mnist_train.csv")
tst_data = pd.read_csv("./Data/sign_mnist_test/sign_mnist_test.csv")

#Grab only the data from data frame and normalize
train_data = tr_data.iloc[:, 1:]/255
train_data = train_data.to_numpy()
train_data = np.reshape(train_data, (27455, 28, 28)) #transform into 27455 (num images) 28x28 2D arrays (turn into pictures)

test_data = tst_data.iloc[:, 1:]/255.0
test_data = test_data.to_numpy()
test_data = np.reshape(test_data, (7172, 28, 28))

#Grab labels from data frame
train_labels = tr_data["label"].to_numpy()
train_labels = np.reshape(train_labels, (27455, 1))

test_labels = tst_data["label"].to_numpy()
test_labels = np.reshape(test_labels, (7172, 1))

#Create model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation="softmax"))

#Compile and train
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

#Get results of training
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(test_acc)

#Save trained model
model.save("ASLmodel.keras")