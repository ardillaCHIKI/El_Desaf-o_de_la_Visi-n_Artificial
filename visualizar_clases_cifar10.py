import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Cargar el dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Nombres de las clases
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Seleccionar una imagen por clase
samples = []
for class_id in range(10):
    idx = np.where(y_train == class_id)[0][0]  # Primer índice de esa clase
    samples.append(x_train[idx])

# Mostrar las imágenes
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(samples[i])
    plt.title(class_names[i])
    plt.axis('off')
plt.suptitle("Ejemplos representativos de las 10 clases CIFAR-10")
plt.tight_layout()
plt.show()
plt.savefig("muestra_cifar10.png")
print("La cuadrícula de imágenes se ha guardado como 'muestra_cifar10.png'. Puedes abrir este archivo para ver las imágenes si no se muestra la ventana gráfica.")

# Comentarios descriptivos
print("\nObservaciones:")
print("- Las imágenes tienen fondos variados: algunos naturales, otros artificiales.")
print("- Las clases como 'automobile' y 'truck' muestran diferencias en color y forma.")
print("- 'Bird' y 'cat' presentan variabilidad en postura y ángulo.")
print("- 'Ship' puede aparecer en mar abierto o en puerto.")
print("- 'Dog' y 'horse' tienen diferencias de raza y tamaño.")
