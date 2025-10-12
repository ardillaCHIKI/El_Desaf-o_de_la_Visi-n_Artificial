import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Cargar y preprocesar datos
(x_train, y_train), (_, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)

# Definir arquitectura CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Mostrar métricas por época
print("\n📊 Métricas por época:")
for i in range(10):
    print(f"Época {i+1}: "
          f"Loss = {history.history['loss'][i]:.4f}, "
          f"Accuracy = {history.history['accuracy'][i]:.4f}, "
          f"Val_Loss = {history.history['val_loss'][i]:.4f}, "
          f"Val_Accuracy = {history.history['val_accuracy'][i]:.4f}")
