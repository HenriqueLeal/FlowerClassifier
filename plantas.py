#150 dados

from sklearn.datasets import load_iris #dataset das iris
from pandas.plotting import scatter_matrix #plotagem
import pandas as pd 
from sklearn.model_selection import train_test_split #separar treinamento de teste
import matplotlib.pyplot as plt #plotagem
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



#função para traduzir o tipo da planta, o predict retonar um array
#então o type[0] é pra pegar a primeira posição do array
def flower_is(type):
   return {
      0: "setosa",
      1: "versicolor",
      2: "virginica"
   }[type[0]]





iris_data = load_iris() #carregou dados
print(iris_data.keys()) #features, variaveis que posso chamar

#print(iris_data["DESCR"])  #descrição

print(iris_data["data"][0:6])  #dados das plantas
print(iris_data["feature_names"]) #nome das características
print(iris_data["target"][0:100])  #index da planta
print(iris_data["target_names"]) #nome das plantas


############# separação de treinamento e teste, 25% e 75%

X_train, X_test, Y_train, Y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0) 
print("quantidade de dados de treino: {}".format(X_train.shape))
print("quantidade de dados de teste: {}".format(X_test.shape))

######################################################


iris_dataframe = pd.DataFrame(X_train, columns=iris_data.feature_names)
grr = scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.plot(iris_dataframe)
#plt.show()


knn = KNeighborsClassifier(n_neighbors = 3) #parametro de vizinhos proximos que ira verificar
knn.fit(X_train, Y_train)  #treino



print("Treino Score: {}".format(knn.score(X_train, Y_train))) #acertos treino
print("Teste Score: {}".format(knn.score(X_test, Y_test))) #acertos teste



#flowerx = np.array([[5.1, 2.7, 1, 0.5]]) #criação de flor para predição setosa
flowerx = np.array([[6.3, 4., 6, 2]]) #virginica



flower_type = flower_is(knn.predict(flowerx))

print("flor da classe: " + str(flower_type))





