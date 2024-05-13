"""
Authors: Evan Scott, Kieran Kennedy, Sean Pala
Last Modified: 5/12/24
Description: Handles the preprocessing of data from ASL dataset to be fit with CNN model
"""
import pandas as pd
import numpy as np

class Data_Configuration:

    @staticmethod
    def config():
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

        return train_data, test_data, train_labels, test_labels
