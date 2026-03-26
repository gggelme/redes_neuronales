import numpy as np

class simple_perceptron():
    def __init__(self, learning_rate = 0.01, max_epochs=1000, batch_size = 1,error_threshold = 0.01, activate_function = 'identity'):
        #implementar correccion de y para la sigmoide
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_threshold = error_threshold
        self.weights = None
        self.bias = None
        self.batch_size = batch_size
        if activate_function == 'identity':
            self.activate_function = self._identity
            self.gradient = self._grad_identity
            self._error_metric = self._mse_error #regresion
        elif activate_function == 'sign':
            self.activate_function = self._sign
            self.gradient = self._grad_sign
            self._error_metric = self._classification_error
        elif activate_function == 'relu':
            self.activate_function = self._relu
            self.gradient = self._grad_relu
            self._error_metric = self._mse_error
        elif activate_function == 'sigmoid':
            self.activate_function = self._sigmoid
            self.gradient = self._grad_sigmoid
        elif activate_function == 'tanh':
            self.activate_function = self._tanh
            self.gradient = self._grad_tanh
            self._error_metric = self._classification_error_tanh
        elif activate_function == 'sigmoid_bipolar':
            self.activate_function = self._sigmoid_bipolar
            self.gradient = self._grad_sigmoid_bipolar
            self._error_metric = self._classification_error_sigmoid_bipolar
        else:
            raise ValueError("Función de activación no válida")

    def fit (self, X, y):
        n_samples, n_features = X.shape
        bias_column = -1*np.ones((n_samples, 1))
        X_bias = np.hstack((bias_column, X)) #primera columna con bias -1

        self.weights = np.zeros((n_features + 1, 1)) #vector columna
        self.epoch_error = []
        for epoch in range(self.max_epochs):
            epoch_errors = []
            for i in range(0, n_samples, self.batch_size):
                #batches
                X_batch = X_bias[i:i+self.batch_size]
                z_batch= X_batch @self.weights
                y_batch_pred = self.activate_function(z_batch)
                y_batch_real = y[i:i+self.batch_size].reshape(-1,1)

                

                batch_error = self._error_metric(y_batch_pred, y_batch_real)
                epoch_errors.append(batch_error)

                #gradiente
                #error
                e_batch = y_batch_pred - y_batch_real
                phi_prime = self.gradient(z_batch)
                grad_vector= e_batch * phi_prime #producto elemento a elemento
            
                self.weights -= 2 * self.learning_rate * X_batch.T @ grad_vector
            current_epoch_error = np.mean(epoch_errors)
            self.epoch_error.append(current_epoch_error)

            if current_epoch_error < self.error_threshold:
                break


    def transform(self, X):
        n_samples, n_features = X.shape
        bias_column = -1* np.ones((n_samples, 1))
        X_bias = np.hstack((bias_column, X))
        return self.activate_function(X_bias@self.weights)
                
    
    def _identity(self, z):
        return z

    def _grad_identity(self, z):
        return np.ones_like(z)
    
    def _relu(self, z):
        return np.maximum(0, z)

    def _grad_relu(self, z):
        return (z > 0).astype(float)
    
    def _sign(self, z):
        return np.where(z >= 0, 1, -1)

    def _grad_sign(self, z):
        return np.ones_like(z)
    

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _grad_sigmoid(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _tanh(self, z):
        return np.tanh(z)

    def _grad_tanh(self, z):
        return 1 - np.tanh(z)**2
    
    def _sigmoid_bipolar(self, z):
        return (2 / (1 + np.exp(-z))) - 1

    def _grad_sigmoid_bipolar(self, z):
        s = self._sigmoid_bipolar(z)
        return 0.5 * (1 + s) * (1 - s)
    
    def score(self, X, y_real):
        y_real = y_real.reshape(-1, 1)
        y_pred = self.transform(X)
        
        #metrica regresion
        if self.activate_function == self._identity:
            mse = np.mean((y_real - y_pred)**2)
            return {"MSE": mse}
        
        # metrica clasificacion
        else:
            # Para clasificación, convertimos la salida continua en etiquetas discretas
            if self.activate_function == self._sigmoid:
                # Umbral 0.5
                labels_pred = (y_pred >= 0.5).astype(int)
                labels_real = (y_real >= 0.5).astype(int)
            else:
                #umbral 0.
                labels_pred = np.where(y_pred >= 0, 1, -1)
                labels_real = np.where(y_real >= 0, 1, -1)
            
            accuracy = np.mean(labels_pred == labels_real)
            return {"Accuracy": accuracy}
        
    def _mse_error(self, y_pred, y_real):
        """MSE para regresión"""
        return np.mean((y_pred - y_real)**2)
    
    def _classification_error(self, y_pred, y_real):
        """Error de clasificación para funciones con salida -1/1"""
        labels_pred = np.where(y_pred >= 0, 1, -1)
        labels_real = np.where(y_real >= 0, 1, -1)
        return np.mean(labels_pred != labels_real)
    
    def _classification_error_sigmoid(self, y_pred, y_real):
        """Error de clasificación para sigmoid (salida 0/1)"""
        labels_pred = (y_pred >= 0.5).astype(int)
        labels_real = (y_real >= 0.5).astype(int)
        return np.mean(labels_pred != labels_real)
    
    def _classification_error_tanh(self, y_pred, y_real):
        """Error de clasificación para tanh y sigmoid bipolar (salida -1/1)"""
        labels_pred = np.where(y_pred >= 0, 1, -1)
        labels_real = np.where(y_real >= 0, 1, -1)
        return np.mean(labels_pred != labels_real)
    
    def _classification_error_sigmoid_bipolar(self, y_pred, y_real):
        """Error de clasificación para sigmoid bipolar (salida -1/1)"""
        # La sigmoid bipolar también tiene rango [-1, 1]
        labels_pred = np.where(y_pred >= 0, 1, -1)
        labels_real = np.where(y_real >= 0, 1, -1)
        return np.mean(labels_pred != labels_real)
    
    
    


    
