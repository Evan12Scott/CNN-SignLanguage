import tensorflow as tf
from tensorflow.keras import layers, models
import Data_Configuration

class Fit_Model:
    
    def build_test_save(file):

        Data_Configuration.config()

        # Access the variables defined in config method
        train_data = Data_Configuration.train_data
        test_data = Data_Configuration.test_data
        train_labels = Data_Configuration.train_labels
        test_labels = Data_Configuration.test_labels

        #Compile and train
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

        #Get results of training
        test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
        print(test_acc)

        #Save trained model
        model.save(file)