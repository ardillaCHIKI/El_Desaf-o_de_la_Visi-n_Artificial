# Issue 7: Primer Bloque Convolucional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

print("=" * 70)
print("ISSUE 7: PRIMER BLOQUE CONVOLUCIONAL")
print("=" * 70)

# ========== INICIALIZAR EL MODELO ==========
print("\n📦 Inicializando modelo Sequential...")
model = Sequential(name="CNN_CIFAR10")

# Añadir capa Input explícita
model.add(Input(shape=(32, 32, 3)))
print("✅ Modelo creado con Input layer\n")

# ========== AÑADIR PRIMERA CAPA CONVOLUCIONAL ==========
print("=" * 70)
print("AÑADIENDO CAPA CONVOLUCIONAL 1:")
print("=" * 70)

model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    name='conv2d_1'
))
print("\n✅ Capa Conv2D añadida correctamente")

# ========== AÑADIR CAPA DE POOLING ==========
print("\n" + "=" * 70)
print("AÑADIENDO CAPA MAXPOOLING2D:")
print("=" * 70)

model.add(MaxPooling2D(
    pool_size=(2, 2),
    name='maxpool_1'
))
print("\n✅ Capa MaxPooling2D añadida correctamente")

# ========== CONSTRUIR EL MODELO PARA ACCEDER A INPUT Y OUTPUT ==========
model.build(input_shape=(None, 32, 32, 3))

# ========== VERIFICAR LA ESTRUCTURA ==========
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO:")
print("=" * 70)
model.summary()

# ========== ANÁLISIS DE DIMENSIONES ==========
print("\n" + "=" * 70)
print("ANÁLISIS DE DIMENSIONES:")
print("=" * 70)

input_tensor = model.inputs[0]
conv_layer = model.layers[0]   # Conv2D
pool_layer = model.layers[1]   # MaxPooling2D

conv_output_shape = conv_layer.output.shape
pool_output_shape = pool_layer.output.shape

print(f"""
1. INPUT (Imagen de entrada):
   Forma: {input_tensor.shape}
   ├─ Altura: 32 píxeles
   ├─ Ancho: 32 píxeles
   └─ Canales: 3 (RGB)

2. DESPUÉS de Conv2D (32 filtros, kernel 3x3):
   Forma: {conv_output_shape}

3. DESPUÉS de MaxPooling2D (pool_size 2x2):
   Forma: {pool_output_shape}
""")

# ========== VERIFICACIONES AUTOMÁTICAS ==========
print("\n" + "=" * 70)
print("VERIFICACIONES AUTOMÁTICAS:")
print("=" * 70)

# Verificar número de capas
assert len(model.layers) == 2, f"❌ Error: Se esperaban 2 capas (Conv2D + MaxPooling), hay {len(model.layers)}"
print("✅ El modelo tiene 2 capas (Conv2D + MaxPooling2D)")

# Verificar tipos de capas
assert isinstance(conv_layer, Conv2D), "❌ Error: Primera capa no es Conv2D"
assert isinstance(pool_layer, MaxPooling2D), "❌ Error: Segunda capa no es MaxPooling2D"
print("✅ Los tipos de capas son correctos")

# Verificar output shapes
assert conv_output_shape == (None, 30, 30, 32), f"❌ Shape de Conv2D incorrecta: {conv_output_shape}"
assert pool_output_shape == (None, 15, 15, 32), f"❌ Shape de MaxPooling incorrecta: {pool_output_shape}"
print("✅ Output shapes correctos para Conv2D y MaxPooling2D")

# Verificar número de parámetros de Conv2D
expected_params = (3 * 3 * 3 * 32) + 32  # (kernel_h * kernel_w * input_channels * filters) + bias
actual_params = conv_layer.count_params()
assert actual_params == expected_params, f"❌ Parámetros incorrectos: {actual_params} vs {expected_params}"
print(f"✅ Número de parámetros de Conv2D: {actual_params} ✓")

# ========== VISUALIZACIÓN CONCEPTUAL ==========
print("\n" + "=" * 70)
print("VISUALIZACIÓN CONCEPTUAL DEL FLUJO:")
print("=" * 70)
print("""
INPUT IMAGE (32×32×3)
       ↓
   [Conv2D: 32 filtros 3×3 + ReLU]
       ↓
   FEATURE MAPS (30×30×32)
       ↓
   [MaxPooling2D: 2×2]
       ↓
   FEATURE MAPS (15×15×32)
""")

print("\n" + "=" * 70)
print("ISSUE 7 COMPLETADO ✅")
print("=" * 70)
