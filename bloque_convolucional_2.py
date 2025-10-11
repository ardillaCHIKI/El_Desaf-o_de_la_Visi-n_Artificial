# Issue 3(F2): Segundo Bloque Convolucional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

print("=" * 70)
print("ISSUE 3(F2): SEGUNDO BLOQUE CONVOLUCIONAL")
print("=" * 70)

# ========== INICIALIZAR EL MODELO ==========
print("\n📦 Inicializando modelo Sequential...")
model = Sequential(name="CNN_CIFAR10")

# Añadir capa Input explícita
model.add(Input(shape=(32, 32, 3)))
print("✅ Modelo creado con Input layer\n")

# ========== PRIMER BLOQUE CONVOLUCIONAL (del Issue 2) ==========
print("=" * 70)
print("AÑADIENDO PRIMER BLOQUE CONVOLUCIONAL:")
print("=" * 70)

model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    name='conv2d_1'
))
print("✅ Conv2D_1 (32 filtros, 3x3) añadida")

model.add(MaxPooling2D(
    pool_size=(2, 2),
    name='maxpool_1'
))
print("✅ MaxPooling_1 (2x2) añadida")

# ========== SEGUNDO BLOQUE CONVOLUCIONAL (NUEVO) ==========
print("\n" + "=" * 70)
print("AÑADIENDO SEGUNDO BLOQUE CONVOLUCIONAL:")
print("=" * 70)

model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu',
    name='conv2d_2'
))
print("✅ Conv2D_2 (64 filtros, 3x3) añadida")

model.add(MaxPooling2D(
    pool_size=(2, 2),
    name='maxpool_2'
))
print("✅ MaxPooling_2 (2x2) añadida")

# ========== CONSTRUIR EL MODELO PARA ACCEDER A INPUT Y OUTPUT ==========
model.build(input_shape=(None, 32, 32, 3))

# ========== VERIFICAR LA ESTRUCTURA ==========
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO:")
print("=" * 70)
model.summary()

# ========== ANÁLISIS DE DIMENSIONES ==========
print("\n" + "=" * 70)
print("ANÁLISIS COMPLETO DE DIMENSIONES:")
print("=" * 70)

input_tensor = model.inputs[0]
conv1_layer = model.layers[0]   # Conv2D_1
pool1_layer = model.layers[1]   # MaxPooling_1
conv2_layer = model.layers[2]   # Conv2D_2
pool2_layer = model.layers[3]   # MaxPooling_2

conv1_output_shape = conv1_layer.output.shape
pool1_output_shape = pool1_layer.output.shape
conv2_output_shape = conv2_layer.output.shape
pool2_output_shape = pool2_layer.output.shape

print(f"""
FLUJO DE TRANSFORMACIÓN DE DIMENSIONES:

1. INPUT (Imagen de entrada):
   Forma: {input_tensor.shape}
   └─ Tamaño espacial: 32×32, Canales: 3 (RGB)

2. BLOQUE 1 - Conv2D_1 (32 filtros, kernel 3x3):
   Forma: {conv1_output_shape}
   └─ Tamaño espacial: 30×30, Canales: 32

3. BLOQUE 1 - MaxPooling_1 (pool_size 2x2):
   Forma: {pool1_output_shape}
   └─ Tamaño espacial: 15×15, Canales: 32
   └─ ⚡ Reducción espacial: 50%

4. BLOQUE 2 - Conv2D_2 (64 filtros, kernel 3x3):
   Forma: {conv2_output_shape}
   └─ Tamaño espacial: 13×13, Canales: 64
   └─ 🔍 Detecta patrones más complejos

5. BLOQUE 2 - MaxPooling_2 (pool_size 2x2):
   Forma: {pool2_output_shape}
   └─ Tamaño espacial: 6×6, Canales: 64
   └─ ⚡ Segunda reducción espacial
""")

# ========== VERIFICACIONES AUTOMÁTICAS ==========
print("\n" + "=" * 70)
print("VERIFICACIONES AUTOMÁTICAS:")
print("=" * 70)

# Verificar número de capas
assert len(model.layers) == 4, f"❌ Error: Se esperaban 4 capas, hay {len(model.layers)}"
print("✅ El modelo tiene 4 capas (2 bloques convolucionales completos)")

# Verificar tipos de capas
assert isinstance(conv1_layer, Conv2D), "❌ Error: Capa 0 no es Conv2D"
assert isinstance(pool1_layer, MaxPooling2D), "❌ Error: Capa 1 no es MaxPooling2D"
assert isinstance(conv2_layer, Conv2D), "❌ Error: Capa 2 no es Conv2D"
assert isinstance(pool2_layer, MaxPooling2D), "❌ Error: Capa 3 no es MaxPooling2D"
print("✅ Los tipos de capas son correctos")

# Verificar output shapes
assert conv1_output_shape == (None, 30, 30, 32), f"❌ Shape de Conv2D_1 incorrecta: {conv1_output_shape}"
assert pool1_output_shape == (None, 15, 15, 32), f"❌ Shape de MaxPooling_1 incorrecta: {pool1_output_shape}"
assert conv2_output_shape == (None, 13, 13, 64), f"❌ Shape de Conv2D_2 incorrecta: {conv2_output_shape}"
assert pool2_output_shape == (None, 6, 6, 64), f"❌ Shape de MaxPooling_2 incorrecta: {pool2_output_shape}"
print("✅ Output shapes correctos para todas las capas")

# Verificar número de filtros
assert conv1_layer.filters == 32, f"❌ Conv2D_1 debe tener 32 filtros, tiene {conv1_layer.filters}"
assert conv2_layer.filters == 64, f"❌ Conv2D_2 debe tener 64 filtros, tiene {conv2_layer.filters}"
print("✅ Número de filtros correcto (32 → 64)")

# Verificar parámetros
conv1_params = conv1_layer.count_params()
conv2_params = conv2_layer.count_params()
print(f"✅ Parámetros Conv2D_1: {conv1_params:,}")
print(f"✅ Parámetros Conv2D_2: {conv2_params:,}")

# ========== COMPARACIÓN ENTRE BLOQUES ==========
print("\n" + "=" * 70)
print("COMPARACIÓN ENTRE BLOQUES:")
print("=" * 70)
print(f"""
BLOQUE 1:
  • Conv2D: 32 filtros (3×3)
  • Parámetros: {conv1_params:,}
  • Output: 15×15×32
  • Función: Detectar patrones básicos (bordes, texturas simples)

BLOQUE 2:
  • Conv2D: 64 filtros (3×3)
  • Parámetros: {conv2_params:,}
  • Output: 6×6×64
  • Función: Combinar patrones básicos en características complejas

PROGRESIÓN:
  • Tamaño espacial: 32×32 → 15×15 → 6×6
  • Profundidad (filtros): 3 → 32 → 64
  • Información: De píxeles a características abstractas
""")

# ========== VISUALIZACIÓN CONCEPTUAL ==========
print("\n" + "=" * 70)
print("VISUALIZACIÓN CONCEPTUAL DEL FLUJO COMPLETO:")
print("=" * 70)
print("""
INPUT IMAGE (32×32×3)
       ↓
   ╔════════════════════╗
   ║ BLOQUE 1           ║
   ║ [Conv2D: 32@3×3]   ║
   ║       ↓            ║
   ║ (30×30×32)         ║
   ║       ↓            ║
   ║ [MaxPool: 2×2]     ║
   ║       ↓            ║
   ║ (15×15×32)         ║
   ╚════════════════════╝
       ↓
   ╔════════════════════╗
   ║ BLOQUE 2           ║
   ║ [Conv2D: 64@3×3]   ║
   ║       ↓            ║
   ║ (13×13×64)         ║
   ║       ↓            ║
   ║ [MaxPool: 2×2]     ║
   ║       ↓            ║
   ║ (6×6×64)           ║
   ╚════════════════════╝
       ↓
   [Próximo: Flatten + Dense...]
""")

# ========== EXPLICACIÓN CONCEPTUAL ==========
print("\n" + "=" * 70)
print("🧠 ¿POR QUÉ AÑADIR UN SEGUNDO BLOQUE CONVOLUCIONAL?")
print("=" * 70)
print("""
1. JERARQUÍA DE CARACTERÍSTICAS:
   • Bloque 1: Detecta patrones SIMPLES (bordes, líneas, texturas básicas)
   • Bloque 2: Combina patrones simples en formas COMPLEJAS
   
   Ejemplo en CIFAR-10:
   - Bloque 1: Detecta "borde horizontal", "borde vertical", "textura suave"
   - Bloque 2: Combina en "ala de avión", "rueda de coche", "ojo de gato"

2. AUMENTO DE CAPACIDAD REPRESENTACIONAL:
   • 32 filtros → 64 filtros
   • Más filtros = más patrones diferentes que puede aprender
   • Mayor capacidad para distinguir entre 10 clases

3. REDUCCIÓN PROGRESIVA DE DIMENSIONES:
   • Tamaño espacial: 32×32 → 15×15 → 6×6
   • A medida que reducimos el tamaño, aumentamos la profundidad
   • Menos píxeles, pero más información semántica

4. CAMPO RECEPTIVO MÁS GRANDE:
   • Cada neurona en Bloque 2 "ve" un área mayor de la imagen original
   • Puede capturar contexto más amplio
   • Mejor para reconocer objetos completos

5. ABSTRACCIÓN GRADUAL:
   • INPUT: Píxeles crudos (información muy local)
   • Bloque 1: Características de bajo nivel
   • Bloque 2: Características de nivel medio
   • Dense layers: Características de alto nivel → Clasificación

6. MEJOR GENERALIZACIÓN:
   • Dos bloques aprenden representaciones más robustas
   • Menos propenso a overfitting que una sola capa grande
   • Más eficiente en parámetros que capas densas
""")

# ========== CRITERIO DE ACEPTACIÓN ==========
print("\n" + "=" * 70)
print("CRITERIO DE ACEPTACIÓN - ISSUE 3(F2):")
print("=" * 70)
print("""
✅ REQUISITOS CUMPLIDOS:

1. Capa Conv2D con 64 filtros:
   ✓ Filtros: 64
   ✓ Kernel: (3, 3)
   ✓ Activación: 'relu'

2. Capa MaxPooling2D:
   ✓ Pool size: (2, 2)

3. Verificación de dimensiones:
   ✓ Tamaño espacial reducido: 15×15 → 6×6
   ✓ Profundidad aumentada: 32 → 64 canales
   ✓ Reducción correcta aplicada

4. Modelo funcional:
   ✓ Dos bloques convolucionales completos
   ✓ Arquitectura lista para añadir capas Dense
   ✓ Sin errores en la construcción
""")

print("\n" + "=" * 70)
print("ISSUE 3(F2) COMPLETADO ✅")
print("=" * 70)
print("\n🎯 El modelo ahora tiene 2 bloques convolucionales funcionales")
print("📊 Total de parámetros entrenables:", f"{model.count_params():,}")
print("🔜 Próximo paso: Añadir capas Flatten y Dense para clasificación")