"""
author: Rohan Dhanraj
email: rdy5674@gmail.com
""" 

from multiPerceptron import MultiPerceptron
from all_utils import *
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,'running_logs.log'), level=logging.INFO, format=logging_str)


def main(dataset, params, epochs, filename, plotFile):
    data = PrepareData(dataset)

    train, valid, test = data.prepare()

    layers = params[0]
    density = params[1]
    activation_functions = params[2]
    model  = MultiPerceptron(layers, density, activation_functions, list(train[0][0].shape))
    history = model.fit(TRAIN=train, EPOCHS=epochs, VALIDATION=valid)
    score = model.evaluate(*test)

    model.save_model(filename)

    plot_history(history, plotFile)


if __name__ == '__main__':
    datasets = ["mnist", "cifar10", "cifar100", "fashion_mnist"]

    dataset = datasets[0]

    layers = 3
    density = (300, 100, 10)
    activation_functions = ("relu", "relu", "softmax")

    params = [layers, density, activation_functions]
    epochs = 30

    filename = f"{dataset}_model.h5"
    plotFile = f"{dataset}_plot"
    try:
        logging.info('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        main(dataset, params, epochs, filename, plotFile)
        logging.info('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Trained <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    except Exception as e:
        logging.exception(e)
