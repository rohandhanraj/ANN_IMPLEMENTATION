import tensorflow as tf
import numpy as np
import logging
import os


class MultiPerceptron:
    def __init__(self, layers, density, acivation_functions, input_shape):
        """MultiPerceptron class is an implementaation of Multi Layer Perceptron
        which consists of at least three layers of nodes:
        an input layer,
        a hidden layer, and
        an output layer.

        Args:
            layers (int): Number of layers to be created.
            density (tuple): Density of each layer except the input layer
            acivation_functions (tuple): Activation Functions to be applied in each layer
            input_shape (list): Shape of matrix available for input layer
        """
        self.n = layers
        self.density = density
        self.funcs = acivation_functions
        self.shape = input_shape
        self.build()

    def build(self):
        """Builds and compiles the Multi Layer Perceptron Model.
        """
        self.LAYERS = [tf.keras.layers.Flatten(input_shape=self.shape, name="inputLayer")] +\
            [tf.keras.layers.Dense(self.density[i], activation=self.funcs[i], name=f"hiddenLayer{i+1}" if i != self.n - 1 else "outputLayer") for i in range(self.n)]
        logging.info("Layers Stacked\n")

        self.model_clf = tf.keras.models.Sequential(self.LAYERS)
        logging.info(f"{self.model_clf.layers}\n{self.model_clf.summary()}")

        LOSS_FUNCTION = "sparse_categorical_crossentropy"
        OPTIMIZER = "SGD"
        METRICS = ["accuracy"]

        self.model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
        logging.info("Model Compiled")
        logging.info("-----"*10)

    def fit(self, TRAIN, EPOCHS, VALIDATION):
        """Training the model

        Args:
            TRAIN (tuple): Dependent and independent training data
            EPOCHS (int): Number of cycles through training data
            VALIDATION (tuple): Dependent and independent validation data

        Returns:
            pd.DataFrame: to DataFrame
        """
        logging.info("Model Training")
        history = self.model_clf.fit(*TRAIN, epochs=EPOCHS, validation_data=VALIDATION)
        logging.info(f"{history}")
        logging.info("======"*10)
        return history
        
    def predict(self, X):
        """Prediction using the model

        Args:
            X (numpy.ndarray): Test Data

        Returns:
            numpy.ndarray: Predicted array of test data
        """
        logging.info("Predicting")
        logging.info("-----"*10)
        y_prob = self.model_clf.predict(X)
        y_pred = np.argmax(y_prob, axis=1)
        logging.info(f"Predictions:\n{y_pred}")
        logging.info("====="*10)
        return y_pred

    def evaluate(self, X, y):
        """Evaluation metrics of the model: Loss and Accuracy

        Args:
            X (numpy.ndarray): Test Dependent Data
            y (numpy.ndarray): Test Independent Data

        Returns:
            numpy.ndarray: Score 
        """
        logging.info("Evaluating Test Data")
        logging.info("-----"*10)
        score = self.model_clf.evaluate(X, y)
        logging.info(f"Test data evaluation Score: {score}")
        logging.info("====="*10)
        return score

    def save_model(self, filename):
        """This saves the model

        Args:
            model (python object): Trained model to 
            filename (str): path to save the Trained model
        """
        logging.info('Saving Trained model')
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok = True) # ONLY CREATE IF MODEL DIRECTORY DOESN'T EXISTS
        filePath = os.path.join(model_dir, filename) #model/filename
        self.model_clf.save(filePath)
        logging.info(f'Saved the model to {model_dir}')
        logging.info("====="*10)