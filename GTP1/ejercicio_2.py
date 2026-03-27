import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.auxiliares import *
from algorithms.simple_perceptron import simple_perceptron




# ejercicio 2: superficie de decision:
def decision(w, x):
    return -w[1]/w[2] * x  + w[0]/w[2]

def update(frame, ax, modelo, line, x_vals):
    w = modelo.weights_history[frame]

    if w[2] == 0:
        return line,

    y_vals = decision(w, x_vals)
    line.set_data(x_vals, y_vals)

    ax.set_title(f"Epoch {frame}")

    return line,


def crear_animacion(X, modelo, y):
    fig, ax = plt.subplots()

    # scatter
    ax.scatter(X[:,0], X[:,1], c=y)

    # ejes
    ax.axhline(0)
    ax.axvline(0)

    # límites (MUY importante)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    x_vals = np.linspace(-10, 10, 100)

    # línea inicial (vacía)
    line, = ax.plot([], [])

    anim = FuncAnimation(
        fig,
        update,
        frames=len(modelo.weights_history),
        fargs=(ax, modelo, line, x_vals),
        interval=200
    )

    plt.show()

    
    


   




def main():

    base_path = os.path.dirname(__file__) #donde esta este archivo
    ruta_train = os.path.join(base_path, '../data/OR_trn.csv')
    ruta_test = os.path.join(base_path, '../data/OR_tst.csv')

    model_path = os.path.join(base_path, '../models/OR_5_desvio.pkl')

    # cargo el modelo y datos
    try:
        modelo = cargar_modelo(model_path)
    except FileNotFoundError:
        print("No se encontró el modelo OR_5_desvio.pkl")
    except Exception as e:
        print(f"{e}: Corre ejercicio 1.")


    


    #----------------------------traemos los datos del or 5% desvio----------------
    X_train_05_OR, y_train_05_OR = cargar_datos_csv(ruta_train)
    X_test_05_OR, y_test_05_OR = cargar_datos_csv(ruta_test)

    # --------------------------traemos los datos del XOR---------------------------

    X_train_XOR, y_train_XOR = cargar_datos_csv(os.path.join(base_path, "../data/XOR_trn.csv"))
    X_test_XOR, y_test_XOR = cargar_datos_csv(os.path.join(base_path, "../data/XOR_tst.csv"))
    
    ruta_archivo_modelo = os.path.join(base_path, "../models/model_XOR_simple.pkl")

    if os.path.exists(ruta_archivo_modelo):
        print(f"modelo ya encontrado")
        model_XOR_simple = cargar_modelo(ruta_archivo_modelo)
    else:
        model_XOR_simple = simple_perceptron(activate_function = "sign")
        model_XOR_simple.fit(X_train_XOR, y_train_XOR)
        guardar_modelo(ruta_archivo_modelo, model_XOR_simple)

    # ------------------------------------------------------------------------------


    # hacemos dibujo
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    positive_OR_5 = np.where(y_test_05_OR==1)[0]
    negative_OR_5 = np.where(y_test_05_OR==-1)[0]

    # ---------------- dibujo OR ---------------------------------

    axs[0].scatter(X_test_05_OR[positive_OR_5,0], X_test_05_OR[positive_OR_5,1], c= 'red')
    axs[0].scatter(X_test_05_OR[negative_OR_5,0], X_test_05_OR[negative_OR_5,1], c= 'blue')
    axs[0].plot(X_test_05_OR[:,0], decision(modelo.weights,X_test_05_OR[:,0]))

    axs[0].axhline(0, c = 'black',linewidth=1.5)
    axs[0].axvline(0, c = 'black',linewidth=1.5)
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    axs[0].set_title('OR')
    axs[0].grid()


     # ---------------- dibujo XOR ---------------------------------

    positive_XOR = np.where(y_test_XOR==1)[0]
    negative_XOR = np.where(y_test_XOR==-1)[0]

    axs[1].scatter(X_test_XOR[positive_XOR,0], X_test_XOR[positive_XOR,1], c= 'red')
    axs[1].scatter(X_test_XOR[negative_XOR,0], X_test_XOR[negative_XOR,1], c= 'blue')
    axs[1].plot(X_test_XOR[:,0], decision(model_XOR_simple.weights,X_test_XOR[:,0]))

    axs[1].axhline(0, c = 'black',linewidth=1.5)
    axs[1].axvline(0, c = 'black',linewidth=1.5)
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    axs[1].grid()
    axs[1].set_title('XOR')

    plt.tight_layout()
    plt.show()



    crear_animacion (X_train_XOR, model_XOR_simple, y_train_XOR)
    print(model_XOR_simple.score(X_test_XOR, y_test_XOR))







    
    

if (__name__ == "__main__"):
    main()
