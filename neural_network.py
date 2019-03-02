import numpy as np


class NeuralNetwork:

    def __init__(self, x, y, learning_rate):
        self.input_vector = x       # ilosc cech probki
        self.output_vector = y      # odpowiedz dla danej probki
        self.lr = learning_rate
        self.layers = []


    def backpropagation(self):
        pass


    def feedforward(self):
        self.layers[0] = self.layers[0](np.dot(self.input_vector, self.self.layers[0].weight_vector))
        for i in range(1,self.layers.len()):
            self.layer[i] = self.layers[i].activation_function(np.dot(self.layers[i-1], self.layers[i].weight_vector))
            

    def append_layer(self,new_layer):
        self.layers.append(new_layer)


class Layer:
    def __init__(self,input_size,layer_size,activation_function):

        self.weight_vector = np.random.randn(input_size, layer_size)
        self.activation_function = activation_function

    def activation_function(self):
        pass

