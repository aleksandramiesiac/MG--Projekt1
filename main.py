"""

Plik generujacy siec neuronowa

"""

import neural_network as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Wczytywanie pliku konfiguracyjnego i tworzenie slownika z danymi konfiguracyjnymi
config = {}
with open("configuration_file.txt", "r") as conf_file:
    for line in conf_file:
        name, val = line.partition("=")[::2]
        config[name.strip()] = val.replace("\n", "")



### Tworzenie sciezek do plikow z danymi
root = "Dane/" + config["problem"].capitalize() + "/" + "data." + config["set_name"]
tail = config["number_of_samples"] + ".csv"

train_file_path = root + ".train." + tail
test_file_path = root + ".test." + tail



### Wczytywanie danych treningowych i testowych
train_set = pd.read_csv(train_file_path)
test_set = pd.read_csv(test_file_path)

if config["problem"] == "classification":
    ## Dane treningowe
    train_set_X = train_set[["x","y"]].values
    train_set_y = train_set["cls"].values

    n = len(train_set_y)
    m = len(np.unique(train_set_y))

    # train_set_y - one-hot-encoding
    a = train_set_y - 1
    b = np.zeros((n, m))
    b[np.arange(n), a] = 1
    train_set_y_ohe = b


    ## Dane testowe
    test_set_X = test_set[["x","y"]].values
    test_set_y = test_set["cls"].values

    n = len(test_set_y)
    m = len(np.unique(test_set_y))

    # train_set_y - one-hot-encoding
    a = test_set_y - 1
    b = np.zeros((n, m))
    b[np.arange(n), a] = 1
    test_set_y_ohe = b

else:
    pass



### Tworzenie sieci neuronowej
NN = nn.NeuralNetwork(config)

### Proces uczenia sieci
if config["problem"] == "classification":
    NN.train(train_set_X, train_set_y_ohe, int(config["batch_size"]), int(config["number_of_iterations"]))
else:
    pass

### Wynik sieci dla zbioru testowego
NN_output = NN.predict(test_set_X)

print("\nPorownanie wyniku sieci z oczekiwaniami (fragment zbioru danych):")
print(NN_output[1:50])
print(list(test_set_y.T[0][1:50]))

print("\nKońcowa wartość błędu sieci: " + str(NN.loss_values[-1]))

### Sprawdzenie wynikow (porownanie z oczekiwanymi)
score = NN.evaluate(NN_output,test_set_y.T[0])
print("\nAccuracy: " + str(score))

# Wykres loss:
plt.plot(NN.loss_values)
plt.ylabel('loss')
plt.show()
