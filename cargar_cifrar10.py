# Importar el dataset CIFAR-10 desde tensorflow.keras.datasets
from tensorflow.keras.datasets import cifar10

# Cargar los datos y dividir en conjuntos de entrenamiento y prueba
# X_train: imágenes de entrenamiento, y_train: etiquetas de entrenamiento
# X_test: imágenes de prueba, y_test: etiquetas de prueba
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Mostrar las dimensiones de los conjuntos
print(f"Dimensiones de X_train: {X_train.shape}")  # (50000, 32, 32, 3)
print(f"Dimensiones de y_train: {y_train.shape}")  # (50000, 1)
print(f"Dimensiones de X_test: {X_test.shape}")    # (10000, 32, 32, 3)
print(f"Dimensiones de y_test: {y_test.shape}")    # (10000, 1)

# Comentario breve sobre cada variable:
# X_train: imágenes de entrenamiento (matriz de 50000 imágenes de 32x32 píxeles RGB)
# y_train: etiquetas de las imágenes de entrenamiento (clase a la que pertenece cada imagen)
# X_test: imágenes de prueba (matriz de 10000 imágenes de 32x32 píxeles RGB)
# y_test: etiquetas de las imágenes de prueba
