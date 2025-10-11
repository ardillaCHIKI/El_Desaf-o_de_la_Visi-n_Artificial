# Issue 5: Confirmar Estructura 2D del Dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Cargar el dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# ========== VERIFICACIONES DE ESTRUCTURA 2D ==========
print("=" * 60)
print("VERIFICACIÓN DE ESTRUCTURA 2D - ISSUE 5")
print("=" * 60)

# 1. Verificar forma del DATASET COMPLETO (no solo una imagen)
print(f"\nForma del dataset completo: {x_train.shape}")
print(f"  → {x_train.shape[0]} imágenes")
print(f"  → Cada imagen: {x_train.shape[1]}x{x_train.shape[2]} píxeles")
print(f"  → {x_train.shape[3]} canales RGB")

# 2. Verificar una imagen individual
ejemplo = x_train[0]
print(f"\nForma de UNA imagen: {ejemplo.shape}")
print(f"Número de dimensiones: {ejemplo.ndim}D")

# 3. VERIFICACIÓN CRÍTICA: Asegurar que NO está aplanada
assert ejemplo.shape == (32, 32, 3), "❌ ERROR: La imagen no tiene estructura 2D"
assert x_train.ndim == 4, "❌ ERROR: El dataset no mantiene estructura 2D"
print("\n✅ VERIFICACIÓN EXITOSA: Estructura 2D preservada")

# Mostrar la imagen
plt.figure(figsize=(6, 6))
plt.imshow(ejemplo)
plt.title(f"Ejemplo de clase: {y_train[0][0]} - Forma: {ejemplo.shape}")
plt.axis('off')
plt.savefig('verificacion_estructura_2d.png', dpi=150, bbox_inches='tight')
plt.show()

# Guardar la imagen para pruebas futuras
np.save("imagen_ejemplo_cnn.npy", ejemplo)
print("\n✅ Imagen guardada como 'imagen_ejemplo_cnn.npy'")

# ========== COMPARACIÓN VISUAL MLP vs CNN ==========
print("\n" + "=" * 60)
print("COMPARACIÓN: MLP vs CNN")
print("=" * 60)

# Mostrar cómo se vería aplanada (para MLP) vs 2D (para CNN)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CNN: Estructura 2D
axes[0].imshow(ejemplo)
axes[0].set_title(f'✅ PARA CNN\nEstructura 2D: {ejemplo.shape}', 
                  fontsize=12, weight='bold', color='green')
axes[0].axis('off')

# MLP: Estructura 1D (visualización)
ejemplo_plano = ejemplo.flatten()
axes[1].plot(ejemplo_plano[:500], linewidth=0.8, color='red')
axes[1].set_title(f'❌ PARA MLP (NO usar aquí)\nEstructura 1D: {ejemplo_plano.shape}', 
                  fontsize=12, weight='bold', color='red')
axes[1].set_xlabel('Índice del píxel')
axes[1].set_ylabel('Valor del píxel')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_mlp_vs_cnn.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📊 Comparación visual guardada como 'comparacion_mlp_vs_cnn.png'")

# ========== COMENTARIO EXPLICATIVO FINAL ==========
print("\n" + "=" * 60)
print("🧠 EXPLICACIÓN DETALLADA:")
print("=" * 60)
print("""
DIFERENCIA CLAVE: ESTRUCTURA DE DATOS PARA CNN vs MLP

PARA CNN (Convolutional Neural Network):
------------------------------------------
  • Estructura: (32, 32, 3) - SE MANTIENE ✅
  • Las imágenes conservan su estructura tridimensional
  • Dimensiones: [altura, ancho, canales]
  
  ¿Por qué es importante?
  • Los filtros convolucionales recorren la imagen espacialmente
  • Detectan patrones locales: bordes, texturas, formas
  • Se aprovecha la relación entre píxeles vecinos
  • Jerarquía de features: bordes → texturas → formas → objetos
  • Reducción progresiva de dimensionalidad con pooling
  • Mucho más eficiente: menos parámetros, mejor generalización
  
PARA MLP (Multi-Layer Perceptron):
-----------------------------------
  • Estructura: (3072,) - SE APLANARÍA con flatten() ❌
  • Se pierde COMPLETAMENTE la información espacial
  • Cada píxel = feature independiente
  
  Problemas:
  • No se aprovechan patrones locales ni vecindad espacial
  • Píxel en (0,0) no tiene relación con píxel en (0,1)
  • Mayor número de parámetros (3072 × hidden_size)
  • Menos eficiente y más propenso a overfitting
  • No es escalable a imágenes grandes

EJEMPLO PRÁCTICO:
-----------------
Imagina detectar un "borde vertical":
  
  CNN: Un filtro 3x3 detecta el patrón local
       [[-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]]
       ✅ Eficiente: 9 parámetros
  
  MLP: Necesitaría aprender la relación entre píxeles
       sin estructura espacial
       ❌ Ineficiente: miles de parámetros

⚠️  REGLA DE ORO PARA CNN:
    ¡NUNCA aplicar reshape() ni flatten() a los datos de entrada!
    Solo se aplana DESPUÉS de las capas convolucionales, 
    antes de las capas fully connected finales.
""")

print("\n" + "=" * 60)
print("✅ ISSUE 5 COMPLETADO")
print("=" * 60)
print("\nResumen de verificaciones:")
print("  ✅ Estructura 2D mantenida: (32, 32, 3)")
print("  ✅ Dataset completo: (50000, 32, 32, 3)")
print("  ✅ Sin flatten() ni reshape()")
print("  ✅ 3 canales RGB preservados")
print("  ✅ Imagen de ejemplo guardada")
print("  ✅ Comparación visual generada")
print("  ✅ Listo para entrenar CNN")
print("=" * 60)