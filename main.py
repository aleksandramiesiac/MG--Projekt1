"""

Plik generujacy siec neuronowa

"""

import neural_network as nn


# Wczytywanie pliku konfiguracyjnego i tworzenie slownika z danymi konfiguracyjnymi
config = {}
with open("configuration_file.txt", "r") as conf_file:
    for line in conf_file:
        name, val = line.partition("=")[::2]
        config[name.strip()] = val.replace("\n", "")


# Tworzenie sieci neuronowej
# NN = nn.NeuralNetwork(x, y, config)