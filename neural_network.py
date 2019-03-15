"""

Plik zawierajacy klase reprezentujaca siec neuronowa
oraz klasę reprezentujaca warstwe sieci

"""

import numpy as np
import activation_functions as af
from numpy import linalg as LA


class NeuralNetwork:
    def __init__(self, config):
        self.input_vector = None      # warstwa wejsciowa
        self.output_vector = None     # oczekiwana odpowiedz dla danej probki
        # self.output = np.zeros(self.output_vector.shape)     # warstwa wyjściowa
        self.number_of_layers = int(config["number_of_layers"])     # liczba warstw ukrytych( bez input i output)
        self.layers = [None] * self.number_of_layers          # warstwy
        self.layer_size = int(config["layer_size"])           # liczba neuronow w warstwie
        self.activation_function = getattr(af, config["activation_function"])  # funkcja aktywacji; domyslnie - sigmoidalna funkcja unipolarna 
        self.learning_rate = float(config["learning_rate"])     # wspolczynnik nauki
        self.problem = config["problem"]                      # problem: klasyfikacja lub regresja
        self.output = None     # TO DO: w przypadku regresji 1, w przypadku klasyfikacji n (gdzie n - liczba klas)
        self.loss_values = []       # zmienna przygotowana do zapisywania zmieniających się wartości funkcji loss


    def add_layers(self, shape):
        self.layers[0] = Layer(shape, self.layer_size)
        for i in range(1, self.number_of_layers):
            # TO DO: dla roznych liczb neuronow w warstwach (wtedy self.layer_size jest tablica)
            # self.layers[i] = Layer(self.layer_size[i-1], self.layer_size[i])
            self.layers[i] = Layer(self.layer_size, self.layer_size)
            # self.layers[i] = Layer(shape, self.layer_size)


    def backpropagation(self):
        d_weights = [0] * (self.number_of_layers + 1)

        d1 = self.loss_function(True)
        d2 = d1 * self.activation_function(self.output.neurons, True)

        idx = self.number_of_layers
        d_weights[idx] = np.dot(self.layers[idx - 1].neurons.T, d2)
        # print(d_weights[idx].shape)
        idx -= 1
        
        if idx > 0:
            temp = np.dot(d2, self.output.weight_vector.T) * self.activation_function(self.layers[idx].neurons, True)
            d_weights[idx] = np.dot(self.layers[idx-1].neurons.T, temp)
            idx -= 1

            for i in range(idx, 0, -1):
                #print("jestem tu")
                temp = np.dot(temp, self.layers[i+1].weight_vector.T) * self.activation_function(self.layers[i].neurons, True)
                d_weights[i] = np.dot(self.layers[i-1].neurons.T, temp)
        
        if self.number_of_layers == 1:
            x = self.output
        else:
            x = self.layers[1]

        temp = np.dot(temp, x.weight_vector.T) * self.activation_function(self.layers[0].neurons, True)
        d_weights[0] = np.dot(self.input_vector.T, temp)

        # zaktualizowanie wag w kazdej warstwie
        for i in range(self.number_of_layers):
            self.layers[i].weight_vector -= d_weights[i] * self.learning_rate
        self.output.weight_vector -= d_weights[self.number_of_layers] * self.learning_rate


    def loss_function(self, derivative = False):
        x = self.output.neurons - self.output_vector
        return 2 * x if derivative else np.power(LA.norm(x), 2)


    def feedforward(self):
        self.layers[0].neurons = self.activation_function(np.dot(self.input_vector, self.layers[0].weight_vector))
        for i in range(1, self.number_of_layers):
            self.layers[i].neurons = self.activation_function(np.dot(self.layers[i-1].neurons, self.layers[i].weight_vector))

        if self.problem == "regression":
            self.output.neurons = af.linear_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))
        else:
            self.output.neurons = self.activation_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))


    def argmax_output(self, matrix):

        new_output = []

        for element in matrix:
            new_output.append(element.argmax(axis=0)+1)

        return new_output



    def train(self, train_set_X, train_set_y, batch_size, number_of_iterations):
        self.add_layers(train_set_X.shape[1])

        n = 1 if self.problem == "regression" else train_set_y.shape[1]

        self.output = Layer(self.layer_size, n)

        # najlepiej podawać losowe kawałki zbioru danych, zamiast wszystkiego naraz albo pojedynczych próbek,
        # ale powinno zadziałać i tak i tak
        # podział pełnego zbioru treningowego na kawałki:

        for i in range(number_of_iterations):
            for j in range(1,int(train_set_X.shape[0]/batch_size)):
                self.input_vector = train_set_X[j* batch_size - batch_size :j * batch_size, :]
                self.output_vector = train_set_y[j* batch_size - batch_size:j * batch_size, :]
                #print(j* batch_size - batch_size)
                #print(j * batch_size)

                self.feedforward()
                self.backpropagation()
            self.loss_values.append(self.loss_function())
            #print("----------------")

        print(self.argmax_output(self.output.neurons))

        #print(self.output.neurons)
        print(self.argmax_output(self.output_vector))




    def predict(self, test_set_x):
        self.input_vector = test_set_x
        self.feedforward()
        return self.argmax_output(self.output.neurons)


    #def evaluate(self):



class Layer:
    def __init__(self, input_size, layer_size):
        self.size = layer_size
        self.neurons = None
        self.weight_vector = np.random.randn(input_size, layer_size)
