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
    model_90_sign_ruta = os.path.join(base_path, 'models/OR_90_sign.pkl')

    model_90_sign = cargar_modelo(model_90_sign_ruta)

    ruta_train_90 = os.path.join(base_path, 'OR_90_trn.csv')
    ruta_test_90 = os.path.join(base_path, 'OR_90_tst.csv')
    X_train_90, y_train_90 = cargar_datos_csv(ruta_train_90)
    X_test_90, y_test_90 = cargar_datos_csv(ruta_test_90)



    model_90_sigmoid = simple_perceptron(
        activate_function = 'sigmoid_bipolar'
    )
    model_90_sigmoid.fit(X_train_90, y_train_90)


    #graficosss

    # --- FIGURA 1: Regiones de Decisión ---
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Comparación de Fronteras: Ruido 90%', fontsize=16)

    modelos = [(model_90_sign, "Función Sign"), 
               (model_90_sigmoid, "Sigmoide Bipolar")]

    for i, (mod, titulo) in enumerate(modelos):
        ax = axs1[i]
        clase_1 = X_test_90[y_test_90.flatten() == 1]
        clase_neg_1 = X_test_90[y_test_90.flatten() == -1]
        ax.scatter(clase_1[:, 0], clase_1[:, 1], color='#3498db', edgecolors='k', label='Clase 1')
        ax.scatter(clase_neg_1[:, 0], clase_neg_1[:, 1], color='#e74c3c', edgecolors='k', label='Clase -1')

        w = mod.weights.flatten()
        x1_dom = np.linspace(X_test_90[:,0].min()-0.5, X_test_90[:,0].max()+0.5, 100)
        x2_img = (w[0]/w[2]) - (w[1]/w[2])*x1_dom
        ax.plot(x1_dom, x2_img, 'g--', linewidth=2)
        
        ax.set_title(f"{titulo}\nScore: {mod.score(X_test_90, y_test_90)}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Mostrando Figura 1 (Regiones). Ciérrala para ver el Error.")
    plt.show() # <--- EL SCRIPT SE DETIENE AQUÍ HASTA QUE CIERRES LA VENTANA

    # --- FIGURA 2: Evolución del Error ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Evolución del Error por Época', fontsize=16)

    axs2[0].plot(model_90_sign.epoch_error, color='#c0392b')
    axs2[0].set_title("Error Sign (Serrucho)")

    axs2[1].plot(model_90_sigmoid.epoch_error, color='#2980b9')
    axs2[1].set_title("Error Sigmoide Bipolar (Continuo)")

    for ax in axs2:
        ax.set_xlabel("Época")
        ax.set_ylabel("Valor de Error")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Mostrando Figura 2 (Error).")
    plt.show() # <--- SE ABRE RECIÉN AHORA





if (__name__=="__main__"):
    main()

