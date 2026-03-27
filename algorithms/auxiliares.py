import os
import sys
import numpy as np
import csv
import pickle

def cargar_datos_csv(ruta_completa):
    " este archivo extrae datos de un csv escrito de la forma X|Y"
    
    X = []
    y = []
    
    with open(ruta_completa, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        primera_fila = next(lector)
        
        # Detectar si la primera fila es header (contiene strings no numéricos)
        es_header = False
        try:
            # Intentar convertir el primer valor a float
            float(primera_fila[0])
            es_header = False
        except ValueError:
            es_header = True
        
        # Procesar según corresponda
        if not es_header:
            # Procesar primera fila como datos
            datos_fila = [float(val) for val in primera_fila]
            X.append(datos_fila[:-1]) 
            y.append(datos_fila[-1])
        
        # Procesar el resto de filas
        for fila in lector:
            datos_fila = [float(val) for val in fila]
            X.append(datos_fila[:-1]) 
            y.append(datos_fila[-1])  
            
    return np.array(X), np.array(y).reshape(-1, 1)


def cargar_modelo(ruta_completa):
    "esta funcion entra a la carpetamodels y busca el archivo por nombre"
   
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_completa}")
        
    with open(ruta_completa, 'rb') as f:
        modelo = pickle.load(f)
    return modelo

def guardar_modelo(ruta_completa, modelo):
    with open(ruta_completa, 'wb') as f:
        pickle.dump(modelo, f)