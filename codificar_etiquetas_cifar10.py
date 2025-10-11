import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Cargar el dataset
(_, y_train), (_, y_test) = cifar10.load_data()

# Transformar etiquetas a one-hot
y_train_oh = to_categorical(y_train, num_classes=10)
y_test_oh = to_categorical(y_test, num_classes=10)

# Verificar forma
print("Forma de y_train codificado:", y_train_oh.shape)  # Esperado: (50000, 10)
print("Forma de y_test codificado:", y_test_oh.shape)    # Esperado: (10000, 10)

# Mostrar ejemplo
print("\nEjemplo:")
print("Etiqueta original:", y_train[0][0])
print("Vector one-hot:", y_train_oh[0])

# Comentario
print("\n🧠 Comentario:")
print("La codificación one-hot convierte cada clase numérica en un vector binario de 10 posiciones,")
print("donde solo una está activa (1) y el resto son ceros. Esto permite que la red neuronal")
print("trate cada clase como independiente y facilita el cálculo de la función de pérdida.")
