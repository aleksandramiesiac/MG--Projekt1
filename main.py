"""

Plik generujacy siec neuronowa

"""

import neural_network as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


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
loss_values_train =[]
test_set = pd.read_csv(test_file_path)
loss_values_test = []

batch_size=int(config["batch_size"])

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
    ## Dane treningowe
    train_set_X = train_set["x"].values.reshape(len(train_set["x"].values),1)
    train_set_y = train_set["y"].values.reshape(len(train_set["y"].values),1)

    ## Dane testowe
    test_set_X = test_set["x"].values.reshape(len(test_set["x"].values),1)
    test_set_y = test_set["y"].values.reshape(len(test_set["y"].values),1)





### Proces uczenia sieci
batch_size=int(config["batch_size"])

if config["problem"] == "classification":
    ### Tworzenie sieci neuronowej
    NN = nn.NeuralNetwork(config,train_set_X.shape[1],train_set_y_ohe.shape[1])
    for i in range(int(config["number_of_iterations"])):
    
        for j in range(1,int(train_set_X.shape[0]/batch_size)):
            NN.train(train_set_X[j* batch_size - batch_size :j * batch_size, :], train_set_y_ohe[j* batch_size - batch_size:j * batch_size, :])

        ### Wynik sieci dla zbioru testowego
        loss_values_train.append(NN.loss_function())
        NN_output = NN.predict(test_set_X)
        x = NN.output.neurons - test_set_y_ohe
        loss_values_test.append(np.power(LA.norm(x),2))

else:
    ### Tworzenie sieci neuronowej
    NN = nn.NeuralNetwork(config,train_set_X.shape[1],train_set_y.shape[1])
    for i in range(int(config["number_of_iterations"])):
    
        for j in range(1,int(train_set_X.shape[0]/batch_size)):
            NN.train(train_set_X[j* batch_size - batch_size :j * batch_size, :], train_set_y[j* batch_size - batch_size:j * batch_size, :])

        ### Wynik sieci dla zbioru testowego
        loss_values_train.append(NN.loss_function())
        NN_output = NN.predict(test_set_X)
        x = NN.output.neurons - test_set_y
        loss_values_test.append(np.power(LA.norm(x),2))

   






print("\nPorownanie wyniku sieci z oczekiwaniami (fragment zbioru danych):")
print(NN_output[1:50])
print(test_set_y[1:50].T)

print("\nKońcowa wartość błędu sieci: " + str(loss_values_test[-1]))

### Sprawdzenie wynikow (porownanie z oczekiwanymi)
score = NN.evaluate(NN_output,test_set_y.T)
print("\nAccuracy: " + str(score))


#Wykres loss:
fig, ax = plt.subplots()
ax.plot(loss_values_train, 'k', label = "Train set error")
ax.plot(loss_values_test, 'k--', label = "Test set error")
lebel = ax.legend(loc ='upper center',shadow =True ,fontsize ='x-large')
#plt.savefig("Porownania/"+ config["problem"].capitalize() + "/"+ config["set_name"]+config["number_of_samples"] +"loss"+".png")
plt.show()
plt.close()

#Wykres podobieństw
if config["problem"]=="regression":
    plt.plot(NN_output.T[0],test_set_X)
    plt.ylabel('testowe')

    plt.plot(test_set_y,test_set_X)
    plt.legend(('Wyestymowane', 'Prawdziwe'),loc='upper right')
    plt.title(config["set_name"]+config["number_of_samples"])

    #plt.savefig("Porownania/"+ config["problem"].capitalize() + "/"+ config["set_name"]+config["number_of_samples"] + "compare.png")
    plt.show()
else:
    plt.subplot(1,2,1)
    plt.title("Wyestymowane")
    plt.scatter(test_set_X[:,0],test_set_X[:,1], c=NN_output)

    plt.subplot(1,2,2)
    plt.title("Prawdziwe")
    plt.scatter(test_set_X[:,0],test_set_X[:,1], c=test_set_y.T)
   # plt.savefig("Porownania/"+ config["problem"].capitalize() + "/"+ config["set_name"]+config["number_of_samples"] + ".png")
    plt.show()


# #zapisywanie wyników do pliku txt
# f = open("Porownania/"+ config["problem"].capitalize() + "/"+"porownanie.txt", "a")

# #f.write("Warstw "+"Neuronow"+" zbior"+" ilosc_probek"+" ilosc_iteracji"+" lost_value"+" Accuracy\n")
# f.write(config["number_of_layers"]+" "+config["layer_size"] +" "+config["set_name"]+" "+config["number_of_samples"]+" "+config["number_of_iterations"]+" "+str(NN.loss_values[-1])+" "+str(score)+"\n")

# f.close()

