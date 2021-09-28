import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

class PrepareData:
    def __init__(self, dataset):
        self.data = dataset
    
    def load_data(self):
        """It loads the data from the datasets available in Keras

        Returns:
            tuple: It returns tuple containing the training set and the testing set.
        """
        logging.info('Loading data from Keras datasets')
        if self.data == "mnist":
            return tf.keras.datasets.mnist.load_data()
        elif self.data == "cifar10":
            return tf.keras.datasets.cifar10.load_data()
        elif self.data == "cifar100":
            return tf.keras.datasets.cifar100.load_data()
        elif self.data == "fashion_mnist":
            return tf.keras.datasets.fashion_mnist.load_data()

    def prepare(self):
        """It is used to split the Dataset.

        Returns:
            tuple: It returns the tuple containing training, validation and test data.
        """
        (X_train_full, y_train_full),(X_test, y_test) = self.load_data()
        
        l = len(X_train_full) // 10
        size = X_train_full[0].max()
        logging.info('Preparing data by splitting full datasets into train, validation and test')
        
        X_valid, X_train = X_train_full[:l] / size, X_train_full[l:] / size
        y_valid, y_train = y_train_full[:l], y_train_full[l:]

        X_test = X_test / size
        logging.info("-----"*10)

        return  (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def plot_history(history, file_name):
    """It plots the fitting history and saves it to a file.

    Args:
        history: Evaluation data during training
        file_name (str): Filename to save the plot to
    """
    logging.info('Plotting training history')
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.show()
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, file_name) # model/filename
    plt.savefig(plotPath)
    logging.info('Saving the plots at {plotPath}')
    logging.info("#####"*15)
