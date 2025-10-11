# Issue 4: Conversión de Etiquetas a One-Hot Encoding
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

print("=" * 70)
print("ISSUE 4: CONVERSIÓN DE ETIQUETAS A ONE-HOT ENCODING")
print("=" * 70)

# Cargar el dataset
(_, y_train), (_, y_test) = cifar10.load_data()

print("\nAntes de la conversión:")
print(f"  • Forma de y_train: {y_train.shape}")
print(f"  • Forma de y_test: {y_test.shape}")
print(f"  • Tipo de etiquetas: Enteros (0-9)")
print(f"  • Ejemplo de etiquetas originales: {y_train[:5].flatten()}")

# Transformar etiquetas a one-hot
y_train_oh = to_categorical(y_train, num_classes=10)
y_test_oh = to_categorical(y_test, num_classes=10)

print("\nDespués de la conversión:")
print(f"  • Forma de y_train_oh: {y_train_oh.shape}  → 50,000 vectores de 10 posiciones")
print(f"  • Forma de y_test_oh:  {y_test_oh.shape}   → 10,000 vectores de 10 posiciones")
print(f"  • Tipo: Vectores binarios (one-hot)")

# Verificar que la conversión es correcta
assert y_train_oh.shape == (50000, 10), "❌ Error en forma de y_train_oh"
assert y_test_oh.shape == (10000, 10), "❌ Error en forma de y_test_oh"
assert np.allclose(y_train_oh.sum(axis=1), 1), "❌ Error: cada vector debe sumar 1"

print("\n✅ Verificación: Conversión correcta")

# Mostrar ejemplos con nombres de clases
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\n" + "-" * 70)
print("EJEMPLOS DE CONVERSIÓN:")
print("-" * 70)
for i in range(5):
    original = y_train[i][0]
    onehot = y_train_oh[i]
    class_name = class_names[original]
    print(f"Clase {original} ({class_name:10s}) → {onehot}")

# Verificar que se mantiene la correspondencia
print("\n" + "-" * 70)
print("VERIFICACIÓN DE CORRESPONDENCIA:")
print("-" * 70)
for i in range(3):
    original_label = y_train[i][0]
    decoded_label = np.argmax(y_train_oh[i])  # Decodificar one-hot
    match = "✅" if original_label == decoded_label else "❌"
    print(f"Original: {original_label} | Decodificado: {decoded_label} {match}")

print("\n✅ La correspondencia se conserva correctamente")

# Comentario explicativo
print("\n" + "=" * 70)
print("🧠 EXPLICACIÓN DEL ONE-HOT ENCODING:")
print("=" * 70)
print("""
¿Qué es One-Hot Encoding?
-------------------------
La codificación one-hot convierte cada clase numérica (0-9) en un vector 
binario de 10 posiciones, donde:
  • Solo UNA posición está activa (valor 1)
  • Las demás 9 posiciones son cero (valor 0)
  • La posición activa corresponde a la clase

Ejemplo:
  Clase 3 (cat) → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                           ↑
                    Posición 3 activa

¿Por qué es necesario?
----------------------
1. EVITA ORDEN IMPLÍCITO: 
   Sin one-hot, el modelo podría interpretar que clase 9 > clase 3
   Con one-hot, todas las clases son tratadas como independientes

2. COMPATIBLE CON SOFTMAX:
   La última capa usa softmax para generar probabilidades
   One-hot permite comparar directamente con la salida

3. FACILITA EL CÁLCULO DE PÉRDIDA:
   categorical_crossentropy compara vectores de probabilidades
   Necesita que las etiquetas también sean vectores

4. INTERPRETACIÓN PROBABILÍSTICA:
   Cada posición representa P(clase_i | imagen)
   Facilita ver qué tan "segura" está la predicción

5. PREVIENE SESGO NUMÉRICO:
   Sin one-hot, números más grandes podrían tener más peso
   Con one-hot, todas las clases tienen igual importancia
""")

print("=" * 70)
print("ISSUE 4 COMPLETADO ✅")
print("=" * 70)