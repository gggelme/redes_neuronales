import numpy as np
import copy

class K_fold():
    def __init__(self, modelo, k):
        self.modelo = modelo
        self.k = k
        self.metrics = []
    
    def cross_val(self, X, y): # Recibe dataset completo
        N = X.shape[0]
        fold_size = N//self.k

        idx = np.arange(N)
        np.random.shuffle(idx)

        for i in range(self.k):
            modelo_copy = copy.deepcopy(self.modelo)
            idx_select = idx[i*fold_size : (i+1)*fold_size-1]

            mask = np.ones(N, dtype=bool)
            mask[idx_select] = False
            
            X_test, y_test = X[idx_select], y[idx_select].reshape(-1,1)
            X_train, y_train = X[mask], y[mask].reshape(-1,1)

            idx = np.arange(X_train.shape[0])
            np.random.shuffle(idx)
            modelo_copy.idx_shuffle = idx

            # Ahora si intercambiamos edades -1 por la media
            edad_media = np.mean(X_train[:,33])
            X_train[X_train[:,33] == -1] = edad_media
            X_test[X_test[:,33] == -1] = edad_media

            # Diferentes escalas de datos en features
            media = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            X_train = (X_train - media) / std
            X_test = (X_test - media) / std

            modelo_copy.fit(X_train, y_train)
            metrica = modelo_copy.score(X_test, y_test)
            self.metrics.append(metrica)
            print(f"K={i}, Métrica: {metrica}")
            
            cm = modelo_copy.confusion_matrix(X_test, y_test)
            print(cm)
            print("Sensibilidad:", cm[0,0]/(cm[0,0] + cm[0,1])) # SENSIBILIDAD

        return np.mean(self.metrics)
