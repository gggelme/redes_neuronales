"""Ejercicio 1: Implemente el entrenamiento y prueba de un perceptr ́on simple con
una cantidad variable de entradas. Se deben tener en cuenta las siguientes
capacidades: a) lectura de los datos de entrenamiento (entradas y salidas)
desde un archivo de texto separado por comas; b) selecci ́on del criterio de
finalizaci ́on del entrenamiento y n ́umero m ́aximo de  ́epocas; c) selecci ́on de
la tasa de aprendizaje; d) prueba mediante archivos de texto con el mismo
formato.
Pruebe el modelo en la resoluci ́on del problema OR, utilizando los archivos
de datos OR trn.csv y OR tst.csv para el entrenamiento y la prueba, respec-
tivamente. Los patrones que se proveen en estos archivos fueron generados a
partir de los puntos (1,1), (1,-1), (-1,1) y (-1,-1), con peque ̃nas desviaciones
aleatorias (<5 %) en torno a  ́estos."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.simple_perceptron import simple_perceptron
from algorithms.auxiliares import *




def main():

    base_path = os.path.dirname(__file__) #donde esta este archivo
    ruta_train = os.path.join(base_path, 'OR_trn.csv')
    ruta_test = os.path.join(base_path, 'OR_tst.csv')
    
    X_train, y_train = cargar_datos_csv(ruta_train)
    X_test, y_test = cargar_datos_csv(ruta_test)

    print(X_test)

    modelo = simple_perceptron(
        learning_rate=0.01,
        max_epochs = 500,
        error_threshold = 0.01,
        batch_size=1,
        activate_function='sign'
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.transform(X_test)

    print(modelo.score(
        X= X_test,
        y_real= y_test
    ))

    carpeta_modelo = os.path.join(os.path.dirname(__file__), 'models')
    ruta_archivo = os.path.join(carpeta_modelo, 'OR_5_desvio.pkl')

    with open(ruta_archivo, 'wb') as f:
        pickle.dump(modelo, f)
    
    print(f"Modelo guardado exitosamente en: {ruta_archivo}")

    

    
if __name__ == "__main__":
    main()

