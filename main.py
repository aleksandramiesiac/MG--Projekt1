"""

Plik generujacy siec neuronowa

"""

import neural_network as nn
import pandas as pd

# Wczytywanie danych treningowych i testowych

# klasyfikacja:
train_set = pd.read_csv('Dane/Classification/data.circles.train.500.csv')
test_set = pd.read_csv('Dane/Classification/data.circles.test.500.csv')

train_set_X = train_set[['x','y']].values
train_set_y = train_set['cls'].values.reshape(train_set_X.shape[0],1)

test_set_X = test_set[['x','y']].values
test_set_y = test_set['cls'].values.reshape(test_set_X.shape[0],1)


#print(train_set_y.head())

# Wczytywanie pliku konfiguracyjnego i tworzenie slownika z danymi konfiguracyjnymi
config = {}
with open("configuration_file.txt", "r") as conf_file:
    for line in conf_file:
        name, val = line.partition("=")[::2]
        config[name.strip()] = val.replace("\n", "")



# Tworzenie sieci neuronowej
NN = nn.NeuralNetwork(config)

# proces uczenia sieci
NN.train(train_set_X, train_set_y, int(config["number_of_iterations"]))

# wynik sieci dla zbioru testowego
NN_output = NN.predict(test_set_X)

# sprawdzenie wynikow (porownanie z oczekiwanymi)
print(NN_output)
