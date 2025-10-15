from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Definición básica del modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 clases
])

# Compilación del modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Confirmación
print("✅ El modelo se ha compilado correctamente.")

# Explicaciones
print("\n🧠 ¿Por qué Adam?")
print("El optimizador Adam combina las ventajas de AdaGrad y RMSProp, ajustando la tasa de aprendizaje automáticamente.")
print("Es eficiente, requiere poca memoria y funciona bien en redes convolucionales con datos visuales como CIFAR-10.")

print("\n🧠 ¿Por qué categorical_crossentropy?")
print("Esta función de pérdida es ideal para clasificación multiclase con etiquetas codificadas en formato one-hot.")
print("Calcula la distancia entre la distribución real (etiquetas) y la predicha por el modelo.")
