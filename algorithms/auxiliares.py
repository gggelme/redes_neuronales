import os
import sys
import numpy as np
import csv
import pickle

from datetime import datetime 

def parse_valor(val):
    val = val.strip()

    if val == "" or val.lower() in ["nan", "na", "null"]:
        return np.nan

    # 🔥 detectar números tipo 2.987.201,50
    if "." in val and "," in val:
        try:
            val = val.replace(".", "").replace(",", ".")
            return float(val)
        except:
            pass

    # intentar float
    try:
        return float(val)
    except ValueError:
        pass

    # intentar fecha (formato dd.mm.yyyy)
    for fmt in ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(val, fmt).timestamp()
        except ValueError:
            continue

    # fallback
    return np.nan

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
            fila = [val for val in fila if val.strip() != ""]

            datos = [parse_valor(val) for val in fila]
            
            if salidas >= 1:
                X.append(datos[:-salidas])
                y.append(datos[-salidas:])
            else:
                X.append(datos)
        
        # Procesar primera fila si no es header
        if not es_header:
            procesar_fila(primera_fila)
        
        # Procesar resto
        for fila in lector:
            procesar_fila(fila)
    
    return X, y


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