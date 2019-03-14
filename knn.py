import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split #Função de dividir dados de treino e teste
from sklearn.neighbors import KNeighborsClassifier #KNN
import numpy as np

X1, Y1 = make_classification(n_samples=150, n_features=2, n_redundant=0, n_informative=2,
	                         n_clusters_per_class=1, random_state=3, class_sep=1.1, n_classes=3)

knn = KNeighborsClassifier(n_neighbors = 1)

x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state = 1)

knn.fit(x_train, y_train)

print("Treino = {}\n".format(knn.score(x_train, y_train)))
print("Teste = {}\n".format(knn.score(x_test, y_test)))