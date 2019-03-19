"""

Plik z funkcjami aktywacji

"""

import numpy as np


def square_function(x, deriative = False):
    """ Kwadratowa funkcja """
    return 0.005 * 2 * x if deriative else 0.005 * np.power(x, 2)


def ReLU(x, deriative = False):
    """ Funkcja relu """
    return 1. * (x > 0) if deriative else x * (x > 0)


def linear_function(x, derivative = False):
    """ Funkcja liniowa """
    return 0.2 if derivative else (0.2) * x


def cut_linear_function(x, derivative = False):
    """ Obcięta funkcja liniowa  """
    wektor = []
    d = []
    for elem in x:
        if elem < -1 :
            wektor.append(-1)
            d.append(0)
        elif elem > 1:
            wektor.append(1)
            d.append(0)
        else:
            wektor.append(elem)
            d.append(1)
    return d if derivative else wektor


def unipolar_threshold_function(x, derivative = False):
    """ Funkcja progowa unipolarna """
    wektor = []
    d = [0] * len(x)
    for elem in x:
        if elem < 0 :
            wektor.append(0)
        else:
            wektor.append(1)
    return d if derivative else wektor


def bipolar_threshold_function(x, derivative = False):
    """ Funkcja progowa bipolarna """
    wektor = []
    d = [0] * len(x)
    for elem in x:
        if elem < 0 :
            wektor.append(-1)
        else:
            wektor.append(1)
    return d if derivative else wektor


def sigmoid_function(x, derivative = False):
    """ Sigmoidalna funkcja unipolarna """
    return x * (1.0 - x) if derivative else 1.0 / (1.0 + np.exp(-x))


def sigmoid_function_bi(x, derivative = False):
    """ Sigmoidalna funkcja bipolarna (tangens hiperboliczny) """
    return sigmoid_function_bi_d(x) if derivative else sigmoid_function_bi_f(x)


def sigmoid_function_bi_f(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def sigmoid_function_bi_d(x):
    return 1 - sigmoid_function_bi_f(x) ** 2


def gaussian_function(x, derivative = False):
    """Funkcja Gaussa"""
    return -2 * x * np.exp(-x ** 2) if derivative else np.exp((-x**2)/2) 
