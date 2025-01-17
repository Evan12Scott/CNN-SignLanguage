"""
Authors: Evan Scott, Kieran Kennedy, Sean Pala
Last Modified: 5/12/24
Description: CNN model with a balanced layered architecture implementation for ASL image classification.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from Data_Configuration import Data_Configuration

class Balanced_Model:
    
    def build_test_save(file):

        # Access the variables defined in config method
        train_data, test_data, train_labels, test_labels = Data_Configuration.config()

        #Create a generally good fit model
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(26, activation="softmax"))
        
        #Compile and train
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

        #Get results of training
        test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
        print(test_acc)

        #Save trained model
        model.save(file)