# Issue 1: Cargar el Dataset CIFAR-10
from tensorflow.keras.datasets import cifar10

print("=" * 70)
print("ISSUE 1: CARGAR EL DATASET CIFAR-10")
print("=" * 70)

# Cargar los datos y dividir en conjuntos de entrenamiento y prueba
# X_train: imágenes de entrenamiento, y_train: etiquetas de entrenamiento
# X_test: imágenes de prueba, y_test: etiquetas de prueba
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("\n✅ Dataset cargado exitosamente!\n")

# Mostrar las dimensiones de los conjuntos
print("Dimensiones de los conjuntos:")
print(f"  • X_train: {X_train.shape}  → 50,000 imágenes de 32x32 píxeles con 3 canales RGB")
print(f"  • y_train: {y_train.shape}  → 50,000 etiquetas de clase")
print(f"  • X_test:  {X_test.shape}   → 10,000 imágenes de 32x32 píxeles con 3 canales RGB")
print(f"  • y_test:  {y_test.shape}   → 10,000 etiquetas de clase")

# Información adicional
print("\nInformación adicional:")
print(f"  • Tipo de datos de imágenes: {X_train.dtype}")
print(f"  • Rango de valores de píxeles: [{X_train.min()}, {X_train.max()}]")
print(f"  • Tipo de datos de etiquetas: {y_train.dtype}")
print(f"  • Clases únicas: {sorted(set(y_train.flatten()))}")

# Verificación de dimensiones esperadas
assert X_train.shape == (50000, 32, 32, 3), "❌ Error en dimensiones de X_train"
assert X_test.shape == (10000, 32, 32, 3), "❌ Error en dimensiones de X_test"
assert y_train.shape == (50000, 1), "❌ Error en dimensiones de y_train"
assert y_test.shape == (10000, 1), "❌ Error en dimensiones de y_test"

print("\n✅ Verificación de dimensiones: CORRECTA")

# Comentario breve sobre cada variable:
print("\n" + "=" * 70)
print("DESCRIPCIÓN DE LAS VARIABLES:")
print("=" * 70)
print("""
- X_train: Imágenes de entrenamiento
  - Matriz de 50,000 imágenes de 32x32 píxeles RGB
  - Forma: (50000, 32, 32, 3)
  
- y_train: Etiquetas de entrenamiento
  - Clase a la que pertenece cada imagen (0-9)
  - Forma: (50000, 1)
  
- X_test: Imágenes de prueba
  - Matriz de 10,000 imágenes de 32x32 píxeles RGB
  - Forma: (10000, 32, 32, 3)
  
- y_test: Etiquetas de prueba
  - Clase a la que pertenece cada imagen de prueba (0-9)
  - Forma: (10000, 1)
""")

print("=" * 70)
print("ISSUE 1 COMPLETADO ✅")
print("=" * 70)