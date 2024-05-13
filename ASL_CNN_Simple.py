import tensorflow as tf
from tensorflow.keras import layers, models
from Data_Configuration import Data_Configuration

class Simple_Model:
    
    def build_test_save(file):

        # Access the variables defined in config method
        train_data, test_data, train_labels, test_labels = Data_Configuration.config()

        #Create simple, underfit model
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(26, activation="softmax"))

        #Compile and train
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

        #Get results of training
        test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
        print(test_acc)

        #Save trained model
        model.save(file)