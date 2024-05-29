import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Datos de ejemplo (cargas y parámetros del sistema)
inputs = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
outputs = np.array([[2], [3], [4], [5]], dtype=float)

# Definir el modelo
model = Sequential([
    Dense(units=1, input_shape=[2])
])

# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenar el modelo
model.fit(inputs, outputs, epochs=500)

# Hacer predicciones
print(model.predict([[5, 6]]))
