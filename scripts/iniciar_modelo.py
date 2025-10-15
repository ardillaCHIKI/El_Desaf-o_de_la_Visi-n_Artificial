# Issue 2(F2): Iniciar el Modelo Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("=" * 70)
print("ISSUE 6: INICIAR EL MODELO SEQUENTIAL")
print("=" * 70)

# Mostrar versión de TensorFlow
print(f"\nVersión de TensorFlow: {tf.__version__}")
print(f"Versión de Keras: {keras.__version__}")

# ========== IMPORTACIONES NECESARIAS ==========
print("\n" + "-" * 70)
print("LIBRERÍAS IMPORTADAS CORRECTAMENTE:")
print("-" * 70)
print("✅ tensorflow")
print("✅ keras")
print("✅ Sequential (para crear el modelo)")
print("✅ Conv2D (capas convolucionales)")
print("✅ MaxPooling2D (capas de pooling)")
print("✅ Flatten (aplanar los mapas de características)")
print("✅ Dense (capas densas/fully connected)")
print("✅ Dropout (regularización, opcional)")

# ========== INICIALIZAR EL MODELO ==========
print("\n" + "-" * 70)
print("INICIALIZANDO MODELO SEQUENTIAL:")
print("-" * 70)

# Crear el modelo secuencial vacío
model = Sequential(name="CNN_CIFAR10")

print("\n✅ Modelo Sequential creado exitosamente")
print(f"   Nombre del modelo: {model.name}")
print(f"   Tipo: {type(model)}")

# ========== VERIFICAR QUE EL MODELO ESTÁ VACÍO ==========
print("\n" + "-" * 70)
print("VERIFICACIÓN DEL MODELO VACÍO:")
print("-" * 70)

# Verificar número de capas
num_layers = len(model.layers)
print(f"\nNúmero de capas: {num_layers}")

if num_layers == 0:
    print("✅ El modelo está vacío y listo para añadir capas")
else:
    print(f"⚠️  El modelo tiene {num_layers} capa(s)")

# ========== MOSTRAR RESUMEN DEL MODELO ==========
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO (model.summary()):")
print("=" * 70)

try:
    model.summary()
except ValueError as e:
    print("\n⚠️  No se puede mostrar summary() porque el modelo está vacío")
    print(f"   Mensaje de error: {e}")
    print("\n✅ Esto es ESPERADO: el modelo no tiene capas todavía")
    print("   El summary() se podrá ver después de añadir las capas")

# ========== INFORMACIÓN SOBRE EL MODELO ==========
print("\n" + "=" * 70)
print("INFORMACIÓN DEL MODELO:")
print("=" * 70)
print(f"""
Nombre del modelo: {model.name}
Tipo de modelo: Sequential (capas apiladas secuencialmente)
Número de capas: {len(model.layers)}
Estado: Vacío, listo para construcción

Próximos pasos:
  1. Añadir capas convolucionales (extractor de características)
  2. Añadir capas de pooling (reducción de dimensionalidad)
  3. Añadir capa Flatten (aplanar a 1D)
  4. Añadir capas densas (clasificador)
  5. Compilar el modelo (optimizer, loss, metrics)
  6. Entrenar con los datos de CIFAR-10
""")

# ========== VERIFICACIÓN ADICIONAL ==========
print("=" * 70)
print("VERIFICACIONES ADICIONALES:")
print("=" * 70)

# Verificar que es un modelo Sequential
assert isinstance(model, Sequential), "❌ Error: El modelo no es Sequential"
print("✅ El modelo es de tipo Sequential")

# Verificar que está vacío
assert len(model.layers) == 0, "❌ Error: El modelo no está vacío"
print("✅ El modelo está vacío (0 capas)")

# Verificar que se puede modificar
assert not model.built, "❌ Error: El modelo ya está construido"
print("✅ El modelo no está construido (se pueden añadir capas)")

print("\n" + "=" * 70)
print("ISSUE 1(F2) COMPLETADO ✅")
print("=" * 70)
print("\n🎯 El modelo Sequential está listo para añadir capas")
