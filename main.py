"""

Plik generujacy siec neuronowa

"""

import neural_network as nn
import pandas as pd
import numpy as np


### Wczytywanie danych treningowych i testowych

## Klasyfikacja:

# Dane treningowe
train_set = pd.read_csv('Dane/Classification/data.circles.train.500.csv')

train_set_X = train_set[['x','y']].values
train_set_y = train_set['cls'].values   #.reshape(train_set_X.shape[0],1)

n = len(train_set_y)
m = len(np.unique(train_set_y))

# train_set_y - one-hot-encoding
a = train_set_y - 1
b = np.zeros((n, m))
b[np.arange(n), a] = 1
train_set_y_ohe = b


# Dane testowe
test_set = pd.read_csv('Dane/Classification/data.circles.test.500.csv')

test_set_X = test_set[['x','y']].values
test_set_y = test_set['cls'].values.reshape(test_set_X.shape[0],1)

n = len(test_set_y)
m = len(np.unique(test_set_y))

# train_set_y - one-hot-encoding
a = test_set_y - 1
b = np.zeros((n, m))
b[np.arange(n), a] = 1
test_set_y_ohe = b



### Wczytywanie pliku konfiguracyjnego i tworzenie slownika z danymi konfiguracyjnymi
config = {}
with open("configuration_file.txt", "r") as conf_file:
    for line in conf_file:
        name, val = line.partition("=")[::2]
        config[name.strip()] = val.replace("\n", "")



### Tworzenie sieci neuronowej
NN = nn.NeuralNetwork(config)

### Proces uczenia sieci
NN.train(train_set_X, train_set_y_ohe, int(config["number_of_iterations"]))

### Wynik sieci dla zbioru testowego
NN_output = NN.predict(test_set_X)

### Sprawdzenie wynikow (porownanie z oczekiwanymi)
print(NN_output)
