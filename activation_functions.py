"""

Plik z funkcjami aktywacji

"""

import numpy as np


def linear_function(x):
    """ Funkcja liniowa """
    return (1/2) * x


def cut_linear_function(x):
    """ Obcięta funkcja liniowa  """
    wektor = []
    for elem in x:
        if elem < -1 :
            wektor.append(-1)
        elif elem > 1:
            wektor.append(1)
        else:
            wektor.append(elem)
    return wektor


# TO DO: przepisac w ten sam sposob pozostale funkcje (3, 4, 6, 7)


def sigmoid_function(x):
    """ Sigmoidalna funkcja unipolarna """
    return 1.0 / (1.0 + np.exp(-x))


# def activation_function(self):
#     """
#     Funkcje aktywcji do wyboru:
#         1.Funkcja liniowa
#         2.Obcięta funkcja liniowa 
#         3.Funkcja progowa unipolarna 
#         4.Funkcja progowa bipolarna 
#         5.Sigmoidalna funkcja unipolarna 
#         6.Sigmoidalna funkcja bipolarna (tangens hiperboliczny) 
#         7.Funkcja Gaussa 
#     """
#     if self.number_of_activation_function == 1:
#         return (1/2)*self.weight_vector

#     elif self.number_of_activation_function == 2:
#         wektor=[]
#         for x in self.weight_vector:
#             if x < -1 :
#                 wektor.append(-1)
#             elif x > 1:
#                 wektor.append(1)
#             else:
#                 wektor.append(x)

#     elif self.number_of_activation_function == 3:
#         wektor=[]
#         for x in self.weight_vector:
#             if x < 0 :
#                 wektor.append(0)
#             else:
#                 wektor.append(1)
    
#     elif self.number_of_activation_function == 4:
#         wektor=[]
#         for x in self.weight_vector:
#             if x < 0 :
#                 wektor.append(-1)
#             else:
#                 wektor.append(1)

#     elif self.number_of_activation_function == 5:      
#         return 1/(1+np.exp(-self.weight_vector))

#     elif self.number_of_activation_function == 6:
#         return (1-np.exp(-self.weight_vector))/(1+np.exp(-self.weight_vector))

#     elif self.number_of_activation_function == 7:
#         return np.exp((-(self.weight_vector)**2)/2)