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
        self.layers[0] = self.layers[0](np.dot(self.input_vector, self.layers[0].weight_vector))
        for i in range(1,self.layers.len()):
            self.layers[i] = self.layers[i].activation_function(np.dot(self.layers[i-1], self.layers[i].weight_vector))
            

    def append_layer(self,new_layer):
        self.layers.append(new_layer)


class Layer:
    def __init__(self,input_size,layer_size, number_of_function):

        self.weight_vector = np.random.randn(input_size, layer_size)
        self.activation_function = activation_function(number_of_function)

    def activation_function(self, number_of_function):
        """
        Funkcje aktywcji do wyboru:
            1.Funkcja liniowa
            2.Obcięta funkcja liniowa 
            3.Funkcja progowa unipolarna 
            4.Funkcja progowa bipolarna 
            5.Sigmoidalna funkcja unipolarna 
            6.Sigmoidalna funkcja bipolarna (tangens hiperboliczny) 
            7.Funkcja Gaussa 
        """
        if number_of_function ==1:
            return (1/2)*self.weight_vector

        elif number_of_function ==2:
            wektor=[]
            for x in self.weight_vector:
                if x < -1 :
                    wektor.append(-1)
                elif x >1:
                    wektor.append(1)
                else:
                    wektor.append(x)

        elif number_of_function ==3:
            wektor=[]
            for x in self.weight_vector:
                if x < 0 :
                    wektor.append(0)
                else:
                    wektor.append(1)
        
        elif number_of_function ==4:
            wektor=[]
            for x in self.weight_vector:
                if x < 0 :
                    wektor.append(-1)
                else:
                    wektor.append(1)

        elif number_of_function ==5:      
            return 1/(1+np.exp(-self.weight_vector))

        elif number_of_function ==6:
            return (1-np.exp(-self.weight_vector))/(1+np.exp(-self.weight_vector))

        elif number_of_function ==7:
            return np.exp((-(self.weight_vector)**2)/2)
        
        else:
            raise Exception("Błędny numer funkcji aktywacji! Wybierz od 1-7")

            