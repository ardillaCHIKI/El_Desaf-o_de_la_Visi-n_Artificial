import numpy as np
from tensorflow.keras.datasets import cifar10

# Cargar el dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar los valores de píxeles
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# Verificar forma
print("Forma de x_train normalizado:", x_train_norm.shape)  # (50000, 32, 32, 3)
print("Forma de x_test normalizado:", x_test_norm.shape)    # (10000, 32, 32, 3)

# Verificar rango de valores
print("Rango de x_train normalizado:", np.min(x_train_norm), "a", np.max(x_train_norm))
print("Rango de x_test normalizado:", np.min(x_test_norm), "a", np.max(x_test_norm))

# Comentario sobre la normalización
print("\n🧠 Comentario:")
print("Normalizar los valores de píxeles entre 0 y 1 ayuda a que la red neuronal converja más rápido,")
print("reduce el riesgo de explosión o desaparición del gradiente, y mejora la estabilidad numérica")
print("durante el entrenamiento. Además, permite que los pesos aprendidos se ajusten de forma más eficiente.")
