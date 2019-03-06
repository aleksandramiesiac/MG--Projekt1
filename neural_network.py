"""

Plik zawierajacy klase reprezentujaca siec neuronowa
oraz klasę reprezentujaca warstwe sieci

"""

import numpy as np
import activation_functions as af


class NeuralNetwork:
    def __init__(self, x, y, config):
        self.input_vector = x       # warstwa wejsciowa
        self.output_vector = y      # oczekiwana odpowiedz dla danej probki
        # self.output = np.zeros(self.output_vector.shape)     # warstwa wyjściowa
        self.number_of_layers = int(config["number_of_layers"])     # liczba warstw ukrytych( bez input i output)
        self.layers = [None] * self.number_of_layers          # warstwy
        self.layer_size = int(config["layer_size"])           # liczba neuronow w warstwie
        self.activation_function = getattr(af, config["activation_function"])  # funkcja aktywacji; domyslnie - sigmoidalna funkcja unipolarna 
        self.learning_rate = int(config["learning_rate"])     # wspolczynnik nauki
        self.problem = config["problem"]                      # problem: klasyfikacja lub regresja
        self.output = Layer(self.layer_size, 1)     # TO DO: w przypadku regresji 1, w przypadku klasyfikacji n (gdzie n - liczba klas ?)

        for i in range(self.number_of_layers):
            self.layers[i] = Layer(self.input_vector.shape[1], self.layer_size)


    def backpropagation(self):
        # rekurencyjne obliczanie wg 'wzoru lancuchowego' (tak jak w test_NN)
        d_weights = []
        d_weights[0]= np.dot(self.input_vector.T ,self.recursive_loss(self.number_of_layers+1))
        for i in range(1,self.number_of_layers+1):
            d_weights[i] = np.dot(self.layers[i-1].neurons.T ,self.recursive_loss(self.number_of_layers+1-i))

        # zaktualizowanie wag w kazdej warstwie
        for i in range(self.number_of_layers):
            self.layers[i].weight_vector += d_weights[i]*self.learning_rate
        self.output.weight_vector += d_weights[-1]*self.learning_rate

    def recursive_loss(self,n, ktory = -1):
        if n==1:
            return 2 * (self.output.neurons - self.output_vector) * self.activation_function(self.output.neurons,True)
        else:
            ktory+=1
            return np.dot(self.recursive_loss(n-1,ktory),self.layers[ktory].weight_vector.T)*self.activation_function(self.layers[ktory].neurons,True)

    def feedforward(self):
        self.layers[0].neurons = self.activation_function(np.dot(self.input_vector, self.layers[0].weight_vector))
        for i in range(1, self.number_of_layers):
            self.layers[i].neurons = self.activation_function(np.dot(self.layers[i-1].neurons, self.layers[i].weight_vector))
        if self.problem == "regression":
            self.output.neurons = af.linear_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))
        else:
            self.output.neurons = self.activation_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))


    # def append_layer(self, new_layer):
    #     self.layers.append(new_layer)

class Layer:
    def __init__(self, input_size, layer_size):
        self.size = layer_size
        self.neurons = None
        self.weight_vector = np.random.randn(input_size, layer_size)
