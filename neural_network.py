"""

Plik zawierajacy klase reprezentujaca siec neuronowa
oraz klasę reprezentujaca warstwe sieci

"""

import numpy as np
import activation_functions as af


class NeuralNetwork:
    def __init__(self, config):
        self.input_vector = None      # warstwa wejsciowa
        self.output_vector = None     # oczekiwana odpowiedz dla danej probki
        # self.output = np.zeros(self.output_vector.shape)     # warstwa wyjściowa
        self.number_of_layers = int(config["number_of_layers"])     # liczba warstw ukrytych( bez input i output)
        self.layers = [None] * self.number_of_layers          # warstwy
        self.layer_size = int(config["layer_size"])           # liczba neuronow w warstwie
        self.activation_function = getattr(af, config["activation_function"])  # funkcja aktywacji; domyslnie - sigmoidalna funkcja unipolarna 
        self.learning_rate = int(config["learning_rate"])     # wspolczynnik nauki
        self.problem = config["problem"]                      # problem: klasyfikacja lub regresja
        self.output = Layer(self.layer_size, 1)     # TO DO: w przypadku regresji 1, w przypadku klasyfikacji n (gdzie n - liczba klas)


    def add_layers(self,shape):
        for i in range(self.number_of_layers):
            self.layers[i] = Layer(shape, self.layer_size)


    def backpropagation(self):

        pass

        # rekurencyjne obliczanie wg 'wzoru lancuchowego' (tak jak w test_NN)
        #d_weights = []

        # ostatnia warstwa:
        #d1 = self.loss_function(True)
        #d2 = d1 * self.output * (1 - self.output)


        #for i in range(1,self.number_of_layers+1,-1):
            #d_weights[i] = np.dot(self.layers[i-1].neurons.T ,self.recursive_loss(self.number_of_layers+1-i))

        # zaktualizowanie wag w kazdej warstwie
        #for i in range(self.number_of_layers):
        #    self.layers[i].weight_vector += d_weights[i]*self.learning_rate
        #self.output.weight_vector += d_weights[-1]*self.learning_rate


    def loss_function(self, derivative = False):
        return 2*(self.output - self.output_vector) if derivative else np.pow(self.output - self.output_vector)

#    def recursive_loss(self,n, ktory = -1):
#        if n==1:
#           print("n: " + str(n))
#            return 2 * (self.output.neurons - self.output_vector) * self.activation_function(self.output.neurons,True)
#        else:
#            print("n: " + str(n))
#            print(self.layers[ktory].weight_vector.T.shape)
#            print("wynik funkcji recursive loss: ")
#            print(self.recursive_loss(n - 1, ktory))
#            print(np.dot(self.recursive_loss(n-1,ktory),self.layers[ktory].weight_vector.T))
#            ktory+=1
#            return np.dot(self.recursive_loss(n-1,ktory),self.layers[ktory].weight_vector.T)*self.activation_function(self.layers[ktory].neurons,True)


    def feedforward(self):
        self.layers[0].neurons = self.activation_function(np.dot(self.input_vector, self.layers[0].weight_vector))
        for i in range(1, self.number_of_layers):
            self.layers[i].neurons = self.activation_function(np.dot(self.layers[i-1].neurons, self.layers[i].weight_vector))
        if self.problem == "regression":
            self.output.neurons = af.linear_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))
        else:
            self.output.neurons = self.activation_function(np.dot(self.layers[self.number_of_layers - 1].neurons, self.output.weight_vector))


    def train(self,train_set_X, train_set_y, number_of_iterations):

        self.add_layers(train_set_X.shape[1])

        for i in range(train_set_X.shape[1]):
            print("shape:")
            print(train_set_X.shape[1])
            print("pojedyncza probka?")
            print(train_set_X[i,:])
            print("odpowiedz pojedynczej probki:")
            print(train_set_y[i])

            # najlepiej podawać losowe kawałki zbioru danych, zamiast wszystkiego naraz albo pojedynczych próbek,
            # ale powinno zadziałać i tak i tak

            self.input_vector = train_set_X[i,:]
            self.output_vector = train_set_y[i]
            for j in range(number_of_iterations):
                self.feedforward()
                print("przeszło feedforward")
                self.backpropagation()


    def predict(self, test_set_x):
        self.input_vector = test_set_x
        self.feedforward()
        return self.output


    #def evaluate(self):



class Layer:
    def __init__(self, input_size, layer_size):
        self.size = layer_size
        self.neurons = None
        self.weight_vector = np.random.randn(input_size, layer_size)
