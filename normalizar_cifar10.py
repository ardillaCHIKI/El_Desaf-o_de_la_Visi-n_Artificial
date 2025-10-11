# Issue 3: Normalización de Imágenes
import numpy as np
from tensorflow.keras.datasets import cifar10

print("=" * 70)
print("ISSUE 3: NORMALIZACIÓN DE IMÁGENES")
print("=" * 70)

# Cargar el dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("\n📊 ANTES de la normalización:")
print(f"  • Forma de x_train: {x_train.shape}")
print(f"  • Forma de x_test:  {x_test.shape}")
print(f"  • Tipo de datos: {x_train.dtype}")
print(f"  • Rango de valores: [{x_train.min()}, {x_train.max()}]")
print(f"  • Media: {x_train.mean():.2f}")
print(f"  • Desviación estándar: {x_train.std():.2f}")

# Normalizar los valores de píxeles (dividir entre 255)
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

print("\n📊 DESPUÉS de la normalización:")
print(f"  • Forma de x_train_norm: {x_train_norm.shape}")
print(f"  • Forma de x_test_norm:  {x_test_norm.shape}")
print(f"  • Tipo de datos: {x_train_norm.dtype}")
print(f"  • Rango de valores: [{x_train_norm.min()}, {x_train_norm.max()}]")
print(f"  • Media: {x_train_norm.mean():.4f}")
print(f"  • Desviación estándar: {x_train_norm.std():.4f}")

# Verificaciones de seguridad
assert x_train_norm.shape == (50000, 32, 32, 3), "❌ Error: forma de x_train cambió"
assert x_test_norm.shape == (10000, 32, 32, 3), "❌ Error: forma de x_test cambió"
assert 0 <= x_train_norm.min() <= x_train_norm.max() <= 1, "❌ Error: valores fuera de rango [0,1]"
assert 0 <= x_test_norm.min() <= x_test_norm.max() <= 1, "❌ Error: valores fuera de rango [0,1]"

print("\n✅ Verificación: Normalización correcta")
print("✅ La forma se mantiene: (32, 32, 3)")
print("✅ Todos los valores están en el rango [0, 1]")

# Comentario sobre la normalización
print("\n" + "=" * 70)
print("🧠 ¿POR QUÉ LA NORMALIZACIÓN MEJORA EL ENTRENAMIENTO?")
print("=" * 70)
print("""
1. CONVERGENCIA MÁS RÁPIDA:
   • Los optimizadores (SGD, Adam) funcionan mejor con valores pequeños
   • La red aprende más rápido cuando los inputs están en escala similar
   • Se necesitan menos épocas para alcanzar buenos resultados

2. ESTABILIDAD NUMÉRICA:
   • Evita valores muy grandes que pueden causar overflow
   • Previene problemas de precisión en operaciones con float32
   • Reduce el riesgo de NaN (Not a Number) durante el entrenamiento

3. PREVIENE GRADIENTES EXPLOSIVOS/DESAPARECIDOS:
   • Valores grandes → gradientes grandes → inestabilidad
   • Valores pequeños y normalizados → gradientes controlados
   • Facilita el flujo de información durante backpropagation

4. EQUILIBRIO ENTRE FEATURES:
   • Todos los píxeles están en la misma escala [0, 1]
   • Ningún píxel domina el aprendizaje por tener valores mayores
   • La red trata todas las features con igual importancia inicial

5. COMPATIBILIDAD CON FUNCIONES DE ACTIVACIÓN:
   • Sigmoid y Tanh saturan con valores grandes (gradiente → 0)
   • ReLU funciona mejor con inputs normalizados
   • Mejora la efectividad de las activaciones

6. MEJOR INICIALIZACIÓN DE PESOS:
   • Los pesos iniciales (Xavier, He) se diseñan para inputs normalizados
   • La inicialización funciona óptimamente con valores en [0, 1] o [-1, 1]
   • Reduce el tiempo de "warm-up" del entrenamiento

7. AJUSTE MÁS FÁCIL DE LEARNING RATE:
   • Con inputs normalizados, el learning rate es más intuitivo
   • No es necesario ajustar tanto el learning rate
   • Mayor estabilidad en el proceso de optimización
""")

print("\n" + "=" * 70)
print("📈 IMPACTO PRÁCTICO:")
print("=" * 70)
print("""
SIN normalización (valores 0-255):
  • Learning rate típico: 0.0001 - 0.00001
  • Épocas para converger: 50-100+
  • Riesgo de inestabilidad: ALTO

CON normalización (valores 0-1):
  • Learning rate típico: 0.001 - 0.01
  • Épocas para converger: 20-50
  • Riesgo de inestabilidad: BAJO
""")

print("=" * 70)
print("ISSUE 3 COMPLETADO ✅")
print("=" * 70)