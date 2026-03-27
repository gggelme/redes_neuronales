import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.auxiliares import *
from algorithms.simple_perceptron import simple_perceptron

import os

def decision(w, x):
    return -w[1]/w[2] * x  + w[0]/w[2]



def main():
    base_path = os.path.dirname(__file__) #donde esta este archivo


    ruta_test_5 = os.path.join(base_path, '../data/OR_tst.csv')
    X_test_5, y_test_5 = cargar_datos_csv(ruta_test_5)
    
    ruta_train_50 = os.path.join(base_path, '../data/OR_50_trn.csv')
    ruta_test_50 = os.path.join(base_path, '../data/OR_50_tst.csv')

    X_train_50, y_train_50 = cargar_datos_csv(ruta_train_50)
    X_test_50, y_test_50 = cargar_datos_csv(ruta_test_50)

    ruta_train_90 = os.path.join(base_path, '../data/OR_90_trn.csv')
    ruta_test_90 = os.path.join(base_path, '../data/OR_90_tst.csv')
    X_train_90, y_train_90 = cargar_datos_csv(ruta_train_90)
    X_test_90, y_test_90 = cargar_datos_csv(ruta_test_90)

    ruta_modelo_5 = os.path.join(base_path, '../models/OR_5_desvio.pkl')

    
    modelo_5 = cargar_modelo(ruta_modelo_5)

    
    
    modelo_50 = simple_perceptron(
        learning_rate=0.01,
        max_epochs = 500,
        error_threshold = 0.01,
        batch_size=1,
        activate_function='sign'
    )

    modelo_50.fit(X_train_50, y_train_50)


    modelo_90 = simple_perceptron(
        learning_rate=0.01,
        max_epochs = 500,
        error_threshold = 0.01,
        batch_size=1,
        activate_function='sign'
    )

    modelo_90.fit(X_train_90, y_train_90)


     # guardar modelo 90 para usar la actividad 4
    carpeta_models = os.path.join(base_path, '../models')

    # Nombre sugerido: OR_90_sign.pkl
    ruta_guardado_50 = os.path.join(carpeta_models, 'OR_50_sign.pkl')
    ruta_guardado_90 = os.path.join(carpeta_models, 'OR_90_sign.pkl')

    guardar_modelo(ruta_guardado_50, modelo_50)
    guardar_modelo(ruta_guardado_90, modelo_90)

    

    print (f"""5 : {modelo_5.score(X_test_5, y_test_5)}
           50: {modelo_50.score(X_test_50, y_test_50)}
           90: {modelo_90.score(X_test_90, y_test_90)}""")
    

    #grafico con gpt decision.

    # --- Configuración de los subplots ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Superficies de Decisión con diferentes niveles de ruido (Test Set)', fontsize=16)

    # Datos para iterar en los gráficos
    config = [
        (X_test_5, y_test_5, modelo_5, "Ruido 5%"),
        (X_test_50, y_test_50, modelo_50, "Ruido 50%"),
        (X_test_90, y_test_90, modelo_90, "Ruido 90%")
    ]

    for i, (X, y, modelo, titulo) in enumerate(config):
        ax = axs[i]
        
        # 1. Puntos por clase
        clase_1 = X[y.flatten() == 1]
        clase_neg_1 = X[y.flatten() == -1]

        ax.scatter(clase_1[:, 0], clase_1[:, 1], color='#3498db', edgecolors='k', label='Clase 1')
        ax.scatter(clase_neg_1[:, 0], clase_neg_1[:, 1], color='#e74c3c', edgecolors='k', label='Clase -1')

        # 2. Recta de decisión
        w = modelo.weights.flatten()
        if w[2] != 0:
            x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            dom = np.linspace(x1_min, x1_max, 100)
            # x2 = (w0 / w2) - (w1 / w2) * x1
            img = (w[0] / w[2]) - (w[1] / w[2]) * dom
            ax.plot(dom, img, color='#2ecc71', linestyle='--', linewidth=2)

        # 3. Estética de cada subplot
        ax.set_title(titulo)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2)
        ax.set_ylim(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # grafico aprendizaje:

    # --- FIGURA 2: Curvas de Aprendizaje (Error por Época) ---
    fig_err, axs_err = plt.subplots(1, 3, figsize=(18, 5))
    fig_err.suptitle('Curvas de Aprendizaje: MSE por Época', fontsize=16)

    # Configuramos los modelos para iterar
    config_err = [
        (modelo_5, "Ruido 5%", "#2980b9"),
        (modelo_50, "Ruido 50%", "#8e44ad"),
        (modelo_90, "Ruido 90%", "#c0392b")
    ]

    for i, (modelo, titulo, color) in enumerate(config_err):
        ax_e = axs_err[i]
        
        # Extraemos el historial de errores
        errores = modelo.epoch_error
        
        # Graficamos
        ax_e.plot(range(len(errores)), errores, color=color, linewidth=2)
        
        # Estética
        ax_e.set_title(f"Convergencia - {titulo}")
        ax_e.set_xlabel("Época")
        ax_e.set_ylabel("MSE (Promedio)")
        ax_e.grid(True, linestyle='--', alpha=0.5)
        
        # Añadimos un texto con la cantidad de épocas final
        ax_e.annotate(f'Épocas totales: {len(errores)}', 
                      xy=(0.05, 0.9), xycoords='axes fraction',
                      bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


   
    
    print(f"Modelo de 90% (Sign) guardado exitosamente en: {ruta_guardado_90}")




    


if (__name__ == "__main__"):
    main()
