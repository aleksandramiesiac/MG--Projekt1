"""

Plik generujacy siec neuronowa

"""

import neural_network as nn
import pandas as pd

# Wczytywanie danych treningowych i testowych

# klasyfikacja:
train_set = pd.read_csv('Dane/Classification/data.circles.train.500.csv')
test_set = pd.read_csv('Dane/Classification/data.circles.test.500.csv')

train_set_X = train_set[['x','y']]
train_set_y = train_set['cls']


# Wczytywanie pliku konfiguracyjnego i tworzenie slownika z danymi konfiguracyjnymi
config = {}
with open("configuration_file.txt", "r") as conf_file:
    for line in conf_file:
        name, val = line.partition("=")[::2]
        config[name.strip()] = val.replace("\n", "")


# Tworzenie sieci neuronowej
NN = nn.NeuralNetwork(train_set_X, train_set_y, config)

# proces uczenia sieci

NN.train()

NN.predict()

