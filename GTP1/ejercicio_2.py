import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.auxiliares import *

import os



# ejercicio 2: superficie de decision:
def decision(w, x):
    return -w[1]/w[2] * x  + w[0]/w[2]


def main():

    base_path = os.path.dirname(__file__) #donde esta este archivo
    ruta_train = os.path.join(base_path, 'OR_trn.csv')
    ruta_test = os.path.join(base_path, 'OR_tst.csv')

    model_path = os.path.join(base_path, 'models/OR_5_desvio.pkl')

    # cargo el modelo y datos
    modelo = cargar_modelo(model_path)
    X_train, y_train = cargar_datos_csv(ruta_train)
    X_test, y_test = cargar_datos_csv(ruta_test)


    dominio = np.linspace(-2, 2, 500)
    imagen = decision(modelo.weights, dominio)

    #puntos por clase
    clase_1 = X_test[y_test.flatten() == 1]
    clase_neg_1 = X_test[y_test.flatten() == -1]

    plt.plot(dominio, imagen, color='#2ecc71', linestyle='--', linewidth=2, label='Frontera de Decisión')

    plt.scatter(clase_1[:, 0], clase_1[:, 1], color='#3498db', marker='o', edgecolors='k', s=50, label='Clase 1 (OR True)')
    plt.scatter(clase_neg_1[:, 0], clase_neg_1[:, 1], color='#e74c3c', marker='s', edgecolors='k', s=50, label='Clase -1 (OR False)')

    plt.title('Problema OR: Dispersión y Frontera de Decisión', fontsize=14)
    plt.xlabel('Entrada $x_1$', fontsize=12)
    plt.ylabel('Entrada $x_2$', fontsize=12)
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)
    plt.xlim(X_test[:, 0].min() - 0.2, X_test[:, 0].max() + 0.2)
    plt.ylim(X_test[:, 1].min() - 0.2, X_test[:, 1].max() + 0.2)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)


    plt.show()
    

if (__name__ == "__main__"):
    main()
