import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Cargar el dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Verificar forma de una imagen
ejemplo = x_train[0]
print("Forma de la imagen de ejemplo:", ejemplo.shape)  # Esperado: (32, 32, 3)

# Mostrar la imagen
plt.imshow(ejemplo)
plt.title(f"Ejemplo de clase: {y_train[0][0]}")
plt.axis('off')
plt.show()

# Guardar la imagen para pruebas futuras
np.save("imagen_ejemplo_cnn.npy", ejemplo)

# Comentario sobre estructura
print("\n🧠 Comentario:")
print("Las imágenes conservan su estructura tridimensional (alto, ancho, canales), lo cual es esencial para las CNN.")
print("A diferencia del preprocesamiento en redes MLP, donde se aplicaba flatten() para convertir la imagen en un vector 1D,")
print("las CNN aprovechan la estructura espacial para detectar patrones locales como bordes, texturas y formas.")
