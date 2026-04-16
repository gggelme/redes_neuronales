import numpy as np


class layer():
    """Clase capa:
    
    atributos de clase:
        n_neuronas, n_inputs, weights, outputs, activate, activate_derivative, deltas"""

    def __init__(self, n_neurons, n_inputs, activation_function):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.weights = np.random.uniform(-0.85, 0.85, (self.n_neurons, self.n_inputs+1))
        self.activate_function_name = activation_function  
        self.deltas = np.zeros((n_neurons,1))

        match activation_function:
            case 'sigmoid':
                self.activate = self.sigmoid # 0,1
                self.activate_derivative = self.sigmoid_derivative
            case 'relu':
                self.activate = self.relu
                self.activate_derivative = self.relu_derivative
            case 'symmetry sigmoid':
                self.activate = self.symmetry_sigmoid # -1,1
                self.activate_derivative = self.symmetry_sigmoid_derivative
            case 'identity':
                self.activate = self.identity
                self.activate_derivative = self.identity_derivative
            case 'sign':
                self.activate = self.sign
                self.activate_derivative = self.sign_derivative
           
    def forward(self, inputs):
        """
        recibe un input con bias incluido como columna
        """

        self.inputs = inputs # SIN BIAS, LO AGREGAMOS AHORA

        input_bias = np.vstack((-1, inputs)) #agrego bias a la entrada
        
        self.outputs = self.activate(self.weights @ input_bias)
        return self.outputs #retorna columna
    

    def calculate_deltas(self, error):
        salida = self.activate_derivative(self.outputs) #aca esta el 1/2 en la derivada
         
        for i in range(salida.shape[0]):
            self.deltas[i] = error[i] * salida[i]

        return self.deltas #retorna columna

    
    
    ##funciones auxiliares
    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v)) #0.1
    
    def sigmoid_derivative(self, v):
        return self.sigmoid(v) * (1 - self.sigmoid(v))
    
    def relu(self, v):
        return np.maximum(0, v)
    def relu_derivative(self, v):
        return np.where(v > 0, 1, 0)
    
    def symmetry_sigmoid(self, v):
        return 2 / (1 + np.exp(-v)) - 1
    def symmetry_sigmoid_derivative(self, v):
        return 0.5 * (1 + self.symmetry_sigmoid(v)) * (1 - self.symmetry_sigmoid(v)) 
    
    def identity(self, v):
        return v
    def identity_derivative(self, v):
        return np.ones_like(v)
    
    def sign(self, v):
        return np.where(v >= 0, 1, -1)
    def sign_derivative(self, v):
        return np.zeros_like(v)
    



class neural_network():
    def __init__(self, layers_config, size_input, max_epoch=1000, learning_rate =0.01, error_threshold=1e-6):
        """"Layers config es una lista de tuplas (n_neurons, activation_function)"""
        self.layers = []
        for i in range(len(layers_config)):
            neurons, activation = layers_config[i]
            self.layers.append(
                layer(
                    n_neurons = neurons,
                    n_inputs = size_input  if i==0 else layers_config[i-1][0], #entrada o cantidad de neuronas anterior
                    activation_function = activation
                )
            )
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        self.epoch_error = []
        self.epoch_classification_error = []

    def fit(self, X, y):
        if y.ndim ==1:
            y = y.reshape(-1,1)
        
        es_clasificacion = False
        if self.layers[-1].activate_function_name in ['sigmoid', 'symmetry sigmoid', 'sign']:
            es_clasificacion = True

        #X = np.hstack((-1*np.ones((X.shape[0], 1)), X))
        for epoch in range(self.max_epoch):
            for i in range(X.shape[0]):
                x_current = np.array(X[i,:]).reshape(-1,1) #no le agregamos todavia el bias
                y_current = np.array(y[i,:]).reshape(-1,1) #pueden ser mas de una salida!!!!!!!
                self.forward_propagation(x_current) #lo pasamos como columna!!!!!!!!!!!!!!!
                self.backward_propagation(y_current)
                self._update_weights()
            epoch_error = self.calculate_error_epoc(X, y)
            if es_clasificacion:
                epoch_classification_error = 1 - self.score(X, y)
                self.epoch_classification_error.append(epoch_classification_error)
            #print(f"Época: {epoch} - Errorcito: {epoch_error}")
            self.epoch_error.append(epoch_error)
            if epoch_error < self.error_threshold:
                break

    def transform(self, X):
        """Retorna las predicciones considerando como"""
        y_pred = []
        for i in range(X.shape[0]):
            x_curr = np.array(X[i]).reshape(-1,1) #hacerlo columna
            y_pred.append(self.forward_propagation(x_curr).flatten())

        return np.array(y_pred)
         

    def score(self, X, y):
        correct = 0
        total = len(X)
        n_salidas = y.shape[1]

        es_clasificacion = False
        if self.layers[-1].activate_function_name in ['sigmoid', 'symmetry sigmoid', 'sign']:
            es_clasificacion = True

        if n_salidas > 1 and es_clasificacion:
            for i in range(total):
                x = X[i].reshape(-1,1)
                output = self.forward_propagation(x)
                # [salida_sigmoide para versicolor, salida_sigmoide para otra, salida_sigmoide final]
                pred_class = np.argmax(output)      
                true_class = np.argmax(y[i])       

                if pred_class == true_class:
                    correct += 1
            # accuracy
            return correct / total
        
        if n_salidas == 1 and es_clasificacion:
            # clasificación binaria con 1 salida
            for i in range(total):
                x = X[i].reshape(-1,1)
                output = self.forward_propagation(x)

                y_pred = output.item()
                y_true = y[i].item()

                # Definimos umbral según activación
                if self.layers[-1].activate_function_name in ['sigmoid']:
                    pred_class = 1 if y_pred >= 0.5 else 0
                elif self.layers[-1].activate_function_name in ['symmetry sigmoid']:
                    pred_class = 1 if y_pred >= 0 else -1
                elif self.layers[-1].activate_function_name in ['sign']:
                    pred_class = 1 if y_pred >= 0 else -1

                if pred_class == y_true:
                    correct += 1

            # accuracy
            return correct / total


        if n_salidas == 1 and not es_clasificacion:
            # regresión → usamos MSE como score
            error = 0.0

            for i in range(total):
                x = X[i].reshape(-1,1)
                output = self.forward_propagation(x)

                y_pred = output.item()
                y_true = y[i].item()

                error += (y_pred - y_true) ** 2

            mse = error / total

            return mse
        
        return None



    def forward_propagation(self, x):
        # recorrer capa por capa hasta llegar a la salida
        #siempre es vector
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.layers[i].forward(x)
            else:
                self.layers[i].forward(self.layers[i-1].outputs)

        return self.layers[-1].outputs
    
    def backward_propagation(self, y):
        #i = recorrer capa anterior, cantidad de entradas de actual
        #j recorrer capa actual
        # k recorrer todas la capa de salida

        for layer_index in range(len(self.layers)-1, -1,-1): #recorro todas las capas
            current_layer = self.layers[layer_index]
            if layer_index == len(self.layers)-1: #capa de salida
                error = current_layer.outputs - y #predicho - real (puede ser mas de una salida)
                current_layer.calculate_deltas(error) 
            else:
                next_layer = self.layers[layer_index + 1] #de aca saco el delta de la capa siguiente
                error_propagado = []

                for i in range(current_layer.n_neurons):  # i = neurona actual
                    suma = 0
                    # Sumar contribuciones de todas las neuronas de la SIGUIENTE capa
                    for j in range(next_layer.deltas.shape[0]):
                        # next_layer.weights[j, i+1] donde:
                        # j: neurona siguiente
                        # i+1: neurona actual (sesgo incluido)
                        suma += next_layer.deltas[j] * next_layer.weights[j, i+1]
                    error_propagado.append(suma)
                
                error_propagado = np.array(error_propagado).reshape(-1, 1)
                current_layer.calculate_deltas(error_propagado)

            
    def _update_weights(self):
        for layer_index in range(len(self.layers)):
            #recorro todas las capas
            current_layer = self.layers[layer_index]
            for j in range(current_layer.n_neurons):
                #recorro todas las neuronas de la capa actual
                for i in range(current_layer.n_inputs+1):
                    #producto punto
                    if i == 0:
                        current_layer.weights[j,i] -= self.learning_rate * current_layer.deltas[j].item() * -1
                    else: 
                        current_layer.weights[j,i] -= self.learning_rate * current_layer.deltas[j].item() * current_layer.inputs[i-1].item()


    def calculate_error_epoc(self, X, y):
        y_pred = []

        #predicciones renglon por renglon
        for i in range(X.shape[0]):
            y_pred.append(self.forward_propagation(X[i,:].reshape(-1,1)).flatten()) #pasar como columna
            

        y_pred = np.array(y_pred)
        error = (y_pred-y)**2
        error = error.sum()/len(error)
        return error
        


                



    
