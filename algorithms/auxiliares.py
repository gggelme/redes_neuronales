import os
import sys
import numpy as np
import csv
import pickle

import numpy as np
import csv

def cargar_datos_csv(ruta_completa, salidas=1):
    """Extrae datos de un csv con formato X | Y(s), permitiendo múltiples salidas"""

    X = []
    y = []
    
    with open(ruta_completa, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        primera_fila = next(lector)
        
        # Detectar header
        try:
            float(primera_fila[0])
            es_header = False
        except ValueError:
            es_header = True
        
        # Función auxiliar para procesar filas
        def procesar_fila(fila):
            datos = [float(val) for val in fila]
            X.append(datos[:-salidas])
            y.append(datos[-salidas:])
        
        # Procesar primera fila si no es header
        if not es_header:
            procesar_fila(primera_fila)
        
        # Procesar resto
        for fila in lector:
            procesar_fila(fila)
    
    return np.array(X), np.array(y)


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