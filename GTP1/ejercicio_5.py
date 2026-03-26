import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.auxiliares import *
from algorithms.simple_perceptron import simple_perceptron

def decision(w, x):
    return -w[1]/w[2] * x  + w[0]/w[2]


def main():
    base_path = os.path.dirname(__file__)
    ruta_train = os.path.join(base_path, 'diabetes_trn.csv')
    ruta_test = os.path.join(base_path, 'diabetes_tst.csv')
    X_train, y_train = cargar_datos_csv(ruta_train)
    X_test, y_test = cargar_datos_csv(ruta_test)


    model = simple_perceptron() # por defecto es regresion

    model.fit(X_train, y_train)
    y_pred = model.transform(X_test)
    print(model.score(X_test, y_test))

    plt.scatter(y_pred, y_test)
    plt.show()




if __name__ == "__main__":
    main()