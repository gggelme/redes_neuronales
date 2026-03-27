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
    model_90_sign_ruta = os.path.join(base_path, '../models/OR_90_sign.pkl')

    model_90_sign = cargar_modelo(model_90_sign_ruta)

    ruta_train_90 = os.path.join(base_path, '../data/OR_90_trn.csv')
    ruta_test_90 = os.path.join(base_path, '../data/OR_90_tst.csv')
    X_train_90, y_train_90 = cargar_datos_csv(ruta_train_90)
    X_test_90, y_test_90 = cargar_datos_csv(ruta_test_90)



    model_90_sigmoid = simple_perceptron(
        activate_function = 'sigmoid_bipolar'
    )
    model_90_sigmoid.fit(X_train_90, y_train_90)


    #graficosss

    # --- FIGURA 1: Regiones de Decisión ---
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Comparación de Regiones: Ruido 90%', fontsize=16)

    modelos = [(model_90_sign, "Función Sign"), 
            (model_90_sigmoid, "Sigmoide Bipolar")]

    for i, (mod, titulo) in enumerate(modelos):
        ax = axs1[i]

        # -------- REGIÓN DE DECISIÓN --------
        x_min, x_max = X_test_90[:,0].min() - 1, X_test_90[:,0].max() + 1
        y_min, y_max = X_test_90[:,1].min() - 1, X_test_90[:,1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = mod.transform(grid)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

        # -------- DATOS --------
        clase_1 = X_test_90[y_test_90.flatten() == 1]
        clase_neg_1 = X_test_90[y_test_90.flatten() == -1]

        ax.scatter(clase_1[:, 0], clase_1[:, 1], color='#3498db', edgecolors='k')
        ax.scatter(clase_neg_1[:, 0], clase_neg_1[:, 1], color='#e74c3c', edgecolors='k')

        # -------- RECTA --------
        w = mod.weights.flatten()
        x1_dom = np.linspace(x_min, x_max, 100)
        x2_img = -w[1]/w[2] * x1_dom + w[0]/w[2]

        ax.plot(x1_dom, x2_img, 'g--', linewidth=2)

        ax.set_title(f"{titulo}\nScore: {mod.score(X_test_90, y_test_90)}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



if (__name__=="__main__"):
    main()

