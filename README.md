# ANN_IMPLEMENTATION
Multi Layer Perceptron repository

## Datasets:
1. `"mnist"`,
2. `"cifar10"`,
3. `"cifar100"`,
4. `"fashion_mnist"`

## Activation Functions:
**Non-Linear Neural Networks Activation Functions**
1. `Sigmoid / Logistic Activation Function`
2. `Tanh Function (Hyperbolic Tangent)`
3. `ReLU Function`
4. `Leaky ReLU Function`
5. `Parametric ReLU Function`
6. `Exponential Linear Units (ELUs) Function`
7. `Softmax Function`
8. `Swish`
9. `Gaussian Error Linear Unit (GELU)`
10. `Scaled Exponential Linear Unit (SELU)`

## How to use this?
### Python code
```python
from multiPerceptron import MultiPerceptron
from all_utils import *

data = PrepareData(<Enter the dataset name to be used>)

train, valid, test = data.prepare()

layers = <Enter number of layers>
density = <Enter a tuple of numbers i.e density for each layer>
activation_functions = <Enter a tuple of strings i.e the activation functions for each layer>

model  = MultiPerceptron(layers, density, activation_functions, list(train[0][0].shape))

history = model.fit(TRAIN=train, EPOCHS=epochs, VALIDATION=valid)

score = model.evaluate(*test)

model.save_model(filename)

plot_history(history, plotFile)

````