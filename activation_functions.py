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

def unipolar_threshold_function(x):
    """Funkcja progowa unipolarna """
    wektor = []
    for elem in x:
        if elem < 0 :
            wektor.append(0)
        else:
            wektor.append(1)
    return wektor

def bipolar_threshold_function(x):
    """Funkcja progowa bipolarna """
    wektor = []
    for elem in x:
        if elem < 0 :
            wektor.append(-1)
        else:
            wektor.append(1)
    return wektor

def sigmoid_function_bi(x):
    """Sigmoidalna funkcja bipolarna (tangens hiperboliczny) """
    return (1-np.exp(-x))/(1+np.exp(-x))

def gaussian_function(x):
    """Funkcja Gaussa"""
    return np.exp((-x**2)/2) 
