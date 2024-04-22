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
                 loss_func_gradient=mse_derivative, categories=[]) -> None:
        self.input_size = input_size
        self.output_size = output_size
        print('input', self.input_size)
        print('output', self.output_size)

        self.hidden_layers=hidden_layers

        self.activation_func = activation_func
        self.activation_func_grad=activation_func_grad

        self.loss_function = loss_func
        self.loss_gradient=loss_func_gradient
        self.categories: List[str] = categories

        self.layer1 = FC_Layer(self.input_size,  self.hidden_layers[0])
        self.layer2 = FC_Layer(self.hidden_layers[0],  self.hidden_layers[1])
        self.layer3 = FC_Layer(self.hidden_layers[1],  self.hidden_layers[2])
        self.layer4 = FC_Layer(self.hidden_layers[2],  self.output_size)

        self.activation1=Activation_Layer(self.activation_func, self.activation_func_grad)
        self.activation2=Activation_Layer(self.activation_func, self.activation_func_grad)
        self.activation3=Activation_Layer(self.activation_func, self.activation_func_grad)

    #def fit(self, train_data: np.ndarray, classes: pd.Series) -> None:
    #    return
    def fit(self, x_train: np.ndarray, y_train: pd.Series, epochs=20, learning_rate=0.01) -> None:
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
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
                #print(y_pred)
                # compute loss (for display purpose only)
                err += self.loss_function(y_true, y_pred)

                ##################################################################################
                # Backward Propagation

                layer4_backward = self.layer4.backward_propagation(self.loss_gradient(y_true, y_pred), learning_rate)
                layer3_backward = self.layer3.backward_propagation(self.activation3.backward_propagation(layer4_backward), learning_rate)
                layer2_backward = self.layer2.backward_propagation(self.activation2.backward_propagation(layer3_backward), learning_rate)
                layer1_backward = self.layer1.backward_propagation(self.activation1.backward_propagation(layer2_backward), learning_rate)

                ##################################################################################

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    #def predict(self, test_data: pd.Series) -> np.ndarray:
    #    return
    def predict(self, input_data: np.ndarray) -> np.ndarray: #str:
        '''
        Returns the category corresponding to the highest weight outputted by the network for categorical classification
        Allows predicting a category based on numerical output
        '''
        return_list = []
        for item in input_data:
            weighted_predictions = self.forward(item)
            return_list.append(self.categories[weighted_predictions.argmax()])
            print(self.categories[weighted_predictions.argmax()])
        return(np.array(return_list))
            
        #return(self.categories[weighted_predictions.argmax()])

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        #########################################################################################
        # forward progragation steps ##
        layer1_output: np.ndarray = self.activation1.forward_progagation(self.layer1.forward_propagation(input_data))
        layer2_output: np.ndarray = self.activation2.forward_progagation(self.layer2.forward_propagation(layer1_output))
        layer3_output: np.ndarray = self.activation3.forward_progagation(self.layer3.forward_propagation(layer2_output))
        pred = self.layer4.forward_propagation(layer3_output)

        #########################################################################################
        return pred








# define the activation layer
class Activation_Layer:
    def __init__(self, activation_function, activation_derivative):
        self.activation = activation_function
        self.activation_derivative = activation_derivative

    # returns the activated input
    def forward_progagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    # return the  input_errordE/dX
    def backward_propagation(self, output_error):
        return self.activation_derivative(self.input) * output_error


# define the basic layer.
class FC_Layer:
    def __init__(self, input_size: int, output_size: int):
        # input_size: number of input neurons
        # output_size: number of output neurons
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: np.ndarray = np.random.rand(self.input_size, self.output_size)-0.5 # 2-D array with input_size rows and output_size columns
        self.bias: np.ndarray = np.random.rand(1, self.output_size)-0.5 # horizontal 1-d array with output_size entries


    # return output for a given input x.   
    def forward_propagation(self, input_x: np.ndarray) -> np.ndarray:
        self.input: np.ndarray = input_x
        return np.dot(self.input, self.weights) + self.bias
    

    # compute dE/dW, dE/dB for a given output_error = dE/dY.
    # return input_error = dE/dX.
    # equation is shown in figures.
    def backward_propagation(self, output_error: float, learning_rate: float) -> float:
        ##########################################################################

        input_error: float = np.dot(output_error, self.weights.transpose())
        weights_gradient: float = np.dot(self.input.transpose(), output_error)
        bias_gradient: float = output_error

        ##########################################################################
        # update parameters based on Gradient Descent.
        self.weights = self.weights - learning_rate * weights_gradient
        self.bias    = self.bias - learning_rate * bias_gradient
        return input_error
