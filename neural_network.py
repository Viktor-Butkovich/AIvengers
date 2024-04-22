import pandas as pd
import numpy as np
from typing import List

# Activation function and its derivative.
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_pred - y_true, 2))

def mse_derivative(y_true, y_pred):
    return 2 *(y_pred - y_true)/y_true.size

class neural_network():
    def __init__(self, input_size, output_size, hidden_layers=[64, 32, 64], activation_func=tanh, activation_func_grad=tanh_derivative, loss_func=mse,
                 loss_func_gradient=mse_derivative, categories=[], train_epochs = 20, learning_rate=0.01) -> None:
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers=hidden_layers

        self.activation_func = activation_func
        self.activation_func_grad=activation_func_grad

        self.loss_function = loss_func
        self.loss_gradient=loss_func_gradient
        self.categories: List[str] = categories
        self.category_frequencies = {}

        self.train_epochs = train_epochs
        self.learning_rate = learning_rate

        self.fc_layers = [FC_Layer(self.input_size, self.hidden_layers[0])]
        self.activation_layers = []

        for index, hidden_layer in enumerate(hidden_layers):
            self.activation_layers.append(Activation_Layer(self.activation_func, self.activation_func_grad))
            if index < len(hidden_layers) - 1:
                self.fc_layers.append(FC_Layer(hidden_layer, hidden_layers[index + 1]))
            else:
                self.fc_layers.append(FC_Layer(hidden_layer, self.output_size))

    def fit(self, x_train: np.ndarray, y_train: pd.Series) -> None:
        # sample dimension first
        samples = len(x_train)
        #unique, counts = np.unique(y_train, return_counts=True)
        #category_frequencies = dict(zip(unique, counts))
        #total = sum(counts)
        #mean = sum(counts) / len(counts)
        # Possibly use relative frequencies of each category to more strongly weight samples with rare categories

        # training loop
        for i in range(self.train_epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                input_data = x_train[j]
                input_data = np.expand_dims(input_data, axis=0)
                y_true = y_train.iloc[j] #y_train[j]
                y_true=np.expand_dims(y_true, axis=0)
                y_true = [float(category == y_true) for category in self.categories]
                #print(y_true)
                y_true = np.array(y_true)
                y_pred = self.forward(input_data)
                # compute loss (for display purpose only)
                err += self.loss_function(y_true, y_pred)

                current_backward = self.fc_layers[-1].backward_propagation(self.loss_gradient(y_true, y_pred), self.learning_rate)
                for index in range(len(self.activation_layers) - 1, -1, -1):
                    current_backward = self.fc_layers[index].backward_propagation(self.activation_layers[index].backward_propagation(current_backward), self.learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, self.train_epochs, err))

    def predict(self, input_data: np.ndarray) -> np.ndarray: #str:
        '''
        Returns the category corresponding to the highest weight outputted by the network for categorical classification
        Allows predicting a category based on numerical output
        '''
        return_list = []
        for item in input_data:
            weighted_predictions = self.forward(item)
            return_list.append(self.categories[weighted_predictions.argmax()])
        return(np.array(return_list))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for index, activation_layer in enumerate(self.activation_layers):
            output = activation_layer.forward_propagation(self.fc_layers[index].forward_propagation(output))
        output = self.fc_layers[-1].forward_propagation(output)
        return output

# define the activation layer
class Activation_Layer:
    def __init__(self, activation_function, activation_derivative):
        self.activation = activation_function
        self.activation_derivative = activation_derivative

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    # return the  input_errordE/dX
    def backward_propagation(self, output_error):
        return self.activation_derivative(self.input) * output_error


# define the basic layer.
class FC_Layer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: np.ndarray = np.random.rand(self.input_size, self.output_size)-0.5 # 2-D array with input_size rows and output_size columns
        self.bias: np.ndarray = np.random.rand(1, self.output_size)-0.5 # horizontal 1-d array with output_size entries
 
    def forward_propagation(self, input_x: np.ndarray) -> np.ndarray:
        self.input: np.ndarray = input_x
        return np.dot(self.input, self.weights) + self.bias

    def backward_propagation(self, output_error: float, learning_rate: float) -> float:
        input_error: float = np.dot(output_error, self.weights.transpose())
        weights_gradient: float = np.dot(self.input.transpose(), output_error)
        bias_gradient: float = output_error

        # update parameters based on Gradient Descent.
        self.weights = self.weights - learning_rate * weights_gradient
        self.bias    = self.bias - learning_rate * bias_gradient
        return input_error
