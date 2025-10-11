# Issue 4(F2): Capas Densas (Clasificador)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

print("=" * 70)
print("ISSUE 4(F2): CAPAS DENSAS (CLASIFICADOR)")
print("=" * 70)

# ========== INICIALIZAR EL MODELO ==========
print("\n📦 Inicializando modelo Sequential...")
model = Sequential(name="CNN_CIFAR10")

# Añadir capa Input explícita
model.add(Input(shape=(32, 32, 3)))
print("✅ Modelo creado con Input layer\n")

# ========== BLOQUES CONVOLUCIONALES (Issues anteriores) ==========
print("=" * 70)
print("AÑADIENDO BLOQUES CONVOLUCIONALES:")
print("=" * 70)

# Bloque 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))
print("✅ Bloque 1: Conv2D(32) + MaxPooling añadido")

# Bloque 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2d_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))
print("✅ Bloque 2: Conv2D(64) + MaxPooling añadido")

# ========== CAPAS DENSAS (NUEVO) ==========
print("\n" + "=" * 70)
print("AÑADIENDO CAPAS DENSAS (CLASIFICADOR):")
print("=" * 70)

# 1. Capa Flatten
print("\n1️⃣ FLATTEN:")
model.add(Flatten(name='flatten'))
print("   ✅ Capa Flatten añadida")
print("   Función: Convierte mapas 2D en vector 1D")

# 2. Capa Dense oculta
print("\n2️⃣ DENSE (Capa Oculta):")
model.add(Dense(64, activation='relu', name='dense_hidden'))
print("   ✅ Capa Dense con 64 neuronas y activación 'relu' añadida")
print("   Función: Aprendizaje de combinaciones no lineales")

# 3. Capa de salida
print("\n3️⃣ DENSE (Capa de Salida):")
model.add(Dense(10, activation='softmax', name='dense_output'))
print("   ✅ Capa Dense con 10 neuronas y activación 'softmax' añadida")
print("   Función: Clasificación en 10 clases (probabilidades)")

# ========== CONSTRUIR EL MODELO ==========
model.build(input_shape=(None, 32, 32, 3))

# ========== RESUMEN DEL MODELO COMPLETO ==========
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO COMPLETO:")
print("=" * 70)
model.summary()

# ========== ANÁLISIS DETALLADO DE DIMENSIONES ==========
print("\n" + "=" * 70)
print("ANÁLISIS DETALLADO DE DIMENSIONES:")
print("=" * 70)

input_tensor = model.inputs[0]
conv1_layer = model.layers[0]
pool1_layer = model.layers[1]
conv2_layer = model.layers[2]
pool2_layer = model.layers[3]
flatten_layer = model.layers[4]
dense_hidden = model.layers[5]
dense_output = model.layers[6]

print(f"""
TRANSFORMACIÓN COMPLETA DE DATOS:

1. INPUT (Imagen RGB):
   Shape: {input_tensor.shape}
   Tipo: Imagen 2D con 3 canales
   
2. Conv2D_1 (32 filtros, 3×3):
   Shape: {conv1_layer.output.shape}
   
3. MaxPooling_1 (2×2):
   Shape: {pool1_layer.output.shape}
   
4. Conv2D_2 (64 filtros, 3×3):
   Shape: {conv2_layer.output.shape}
   
5. MaxPooling_2 (2×2):
   Shape: {pool2_layer.output.shape}
   
6. 🔄 FLATTEN (Conversión 2D → 1D):
   Shape: {flatten_layer.output.shape}
   Cálculo: 6 × 6 × 64 = {6*6*64} valores
   ⚠️  PUNTO CRÍTICO: Aquí pasamos de estructura espacial a vector
   
7. Dense_Hidden (64 neuronas, ReLU):
   Shape: {dense_hidden.output.shape}
   Función: Combinar features extraídas
   
8. Dense_Output (10 neuronas, Softmax):
   Shape: {dense_output.output.shape}
   Función: Probabilidades de las 10 clases
""")

# ========== VERIFICACIONES AUTOMÁTICAS ==========
print("\n" + "=" * 70)
print("VERIFICACIONES AUTOMÁTICAS:")
print("=" * 70)

# Verificar número total de capas
assert len(model.layers) == 7, f"❌ Error: Se esperaban 7 capas, hay {len(model.layers)}"
print("✅ El modelo tiene 7 capas (estructura completa)")

# Verificar tipos de capas
assert isinstance(flatten_layer, Flatten), "❌ Error: Capa 4 no es Flatten"
assert isinstance(dense_hidden, Dense), "❌ Error: Capa 5 no es Dense"
assert isinstance(dense_output, Dense), "❌ Error: Capa 6 no es Dense"
print("✅ Tipos de capas correctos (Flatten + 2 Dense)")

# Verificar dimensiones de salida
assert flatten_layer.output.shape == (None, 2304), f"❌ Shape de Flatten incorrecta: {flatten_layer.output.shape}"
assert dense_hidden.output.shape == (None, 64), f"❌ Shape de Dense_Hidden incorrecta: {dense_hidden.output.shape}"
assert dense_output.output.shape == (None, 10), f"❌ Shape de Dense_Output incorrecta: {dense_output.output.shape}"
print("✅ Output shapes correctos para todas las capas densas")

# Verificar activaciones
assert dense_hidden.activation.__name__ == 'relu', "❌ Dense_Hidden debe usar 'relu'"
assert dense_output.activation.__name__ == 'softmax', "❌ Dense_Output debe usar 'softmax'"
print("✅ Funciones de activación correctas (relu y softmax)")

# Verificar número de neuronas
assert dense_hidden.units == 64, f"❌ Dense_Hidden debe tener 64 neuronas, tiene {dense_hidden.units}"
assert dense_output.units == 10, f"❌ Dense_Output debe tener 10 neuronas, tiene {dense_output.units}"
print("✅ Número de neuronas correcto (64 y 10)")

# ========== ANÁLISIS DE PARÁMETROS ==========
print("\n" + "=" * 70)
print("ANÁLISIS DE PARÁMETROS:")
print("=" * 70)

total_params = model.count_params()
conv1_params = conv1_layer.count_params()
conv2_params = conv2_layer.count_params()
dense_hidden_params = dense_hidden.count_params()
dense_output_params = dense_output.count_params()

print(f"""
DISTRIBUCIÓN DE PARÁMETROS:

Capas Convolucionales:
  • Conv2D_1 (32 filtros):     {conv1_params:>8,} parámetros
  • Conv2D_2 (64 filtros):     {conv2_params:>8,} parámetros
  • Subtotal Conv:             {conv1_params + conv2_params:>8,} parámetros

Capas Densas:
  • Dense_Hidden (64 units):   {dense_hidden_params:>8,} parámetros
  • Dense_Output (10 units):   {dense_output_params:>8,} parámetros
  • Subtotal Dense:            {dense_hidden_params + dense_output_params:>8,} parámetros

TOTAL:                         {total_params:>8,} parámetros

💡 Observación: Las capas densas tienen MUCHOS más parámetros
   Dense_Hidden: 2304 × 64 + 64 = {dense_hidden_params:,}
   Dense_Output: 64 × 10 + 10 = {dense_output_params:,}
""")

# ========== VISUALIZACIÓN DE ARQUITECTURA COMPLETA ==========
print("\n" + "=" * 70)
print("ARQUITECTURA COMPLETA DEL MODELO:")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────┐
│                    INPUT (32×32×3)                      │
└─────────────────────────────────────────────────────────┘
                          ↓
    ╔═══════════════════════════════════════════════╗
    ║         EXTRACTOR DE CARACTERÍSTICAS          ║
    ║              (Capas Convolucionales)          ║
    ╠═══════════════════════════════════════════════╣
    ║                                               ║
    ║  ┌─────────────────────────────────────┐     ║
    ║  │ BLOQUE 1                            │     ║
    ║  │  • Conv2D(32, 3×3) + ReLU           │     ║
    ║  │  • MaxPooling(2×2)                  │     ║
    ║  │  → Output: 15×15×32                 │     ║
    ║  └─────────────────────────────────────┘     ║
    ║                    ↓                          ║
    ║  ┌─────────────────────────────────────┐     ║
    ║  │ BLOQUE 2                            │     ║
    ║  │  • Conv2D(64, 3×3) + ReLU           │     ║
    ║  │  • MaxPooling(2×2)                  │     ║
    ║  │  → Output: 6×6×64                   │     ║
    ║  └─────────────────────────────────────┘     ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
                          ↓
    ╔═══════════════════════════════════════════════╗
    ║              CLASIFICADOR                     ║
    ║             (Capas Densas)                    ║
    ╠═══════════════════════════════════════════════╣
    ║                                               ║
    ║  • Flatten: 6×6×64 → 2304                     ║
    ║                    ↓                          ║
    ║  • Dense(64) + ReLU                           ║
    ║                    ↓                          ║
    ║  • Dense(10) + Softmax                        ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
                          ↓
┌─────────────────────────────────────────────────────────┐
│           OUTPUT: Probabilidades (10 clases)            │
│    [P(avión), P(auto), P(pájaro), ..., P(camión)]      │
└─────────────────────────────────────────────────────────┘
""")

# ========== EXPLICACIÓN CONCEPTUAL ==========
print("\n" + "=" * 70)
print("🧠 EXPLICACIÓN DE LAS CAPAS DENSAS:")
print("=" * 70)
print("""
1. ¿POR QUÉ FLATTEN?
   ────────────────
   • Las capas convolucionales trabajan con datos 2D (mapas de características)
   • Las capas Dense solo aceptan datos 1D (vectores)
   • Flatten "aplana" 6×6×64 = 2,304 valores en un vector largo
   
   Ejemplo visual:
   Antes:  [[[1,2], [3,4]], [[5,6], [7,8]]]  (estructura 2D)
   Después: [1, 2, 3, 4, 5, 6, 7, 8]          (vector 1D)

2. ¿POR QUÉ DENSE CON 64 NEURONAS?
   ──────────────────────────────
   • Capa intermedia para combinar features extraídas
   • 64 neuronas es suficiente para CIFAR-10 (no muy complejo)
   • ReLU añade no-linealidad para aprender patrones complejos
   • Actúa como "integrador" de información espacial

3. ¿POR QUÉ DENSE CON 10 NEURONAS Y SOFTMAX?
   ──────────────────────────────────────────
   • 10 neuronas = 10 clases de CIFAR-10
   • Cada neurona representa una clase específica
   • Softmax convierte scores en probabilidades que suman 1.0
   
   Ejemplo de salida:
   [0.7, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
    ↑
   Clase 0 (avión) tiene 70% de probabilidad

4. FLUJO DE INFORMACIÓN:
   ─────────────────────
   • Bloques Conv: "¿Qué features hay en la imagen?"
   • Flatten: "Convirtamos todo en un vector"
   • Dense(64): "¿Cómo se combinan estas features?"
   • Dense(10): "¿A qué clase pertenece?"

5. ¿POR QUÉ NO MÁS CAPAS DENSAS?
   ──────────────────────────────
   • CIFAR-10 es relativamente simple (10 clases, 32×32)
   • Más capas densas = más parámetros = riesgo de overfitting
   • Las capas Conv ya hicieron el trabajo pesado
   • Una capa Dense intermedia es suficiente
""")

# ========== CRITERIO DE ACEPTACIÓN ==========
print("\n" + "=" * 70)
print("CRITERIO DE ACEPTACIÓN - ISSUE 4(F2):")
print("=" * 70)
print("""
✅ REQUISITOS CUMPLIDOS:

1. Capa Flatten:
   ✓ Añadida correctamente
   ✓ Convierte (6, 6, 64) → (2304,)

2. Capa Dense oculta:
   ✓ 64 neuronas
   ✓ Activación 'relu'
   ✓ Parámetros: 2304 × 64 + 64 = 147,520

3. Capa de salida Dense:
   ✓ 10 neuronas (una por clase)
   ✓ Activación 'softmax'
   ✓ Parámetros: 64 × 10 + 10 = 650

4. Resumen del modelo:
   ✓ model.summary() muestra todas las capas
   ✓ Dimensiones correctas en cada etapa
   ✓ Total de parámetros calculado correctamente

5. Modelo completo y funcional:
   ✓ 7 capas en total
   ✓ Arquitectura: Input → Conv → Pool → Conv → Pool → Flatten → Dense → Dense
   ✓ Input shape: (32, 32, 3)
   ✓ Output shape: (10,)
   ✓ Listo para compilar y entrenar
""")

# ========== INFORMACIÓN ADICIONAL ==========
print("\n" + "=" * 70)
print("INFORMACIÓN ADICIONAL:")
print("=" * 70)
print(f"""
📊 Estadísticas del Modelo:
   • Total de capas: {len(model.layers)}
   • Total de parámetros: {total_params:,}
   • Input shape: (32, 32, 3)
   • Output shape: (10,)
   
🎯 Clases CIFAR-10:
   0: airplane    5: dog
   1: automobile  6: frog
   2: bird        7: horse
   3: cat         8: ship
   4: deer        9: truck

📝 Próximos pasos:
   1. Compilar el modelo (optimizer, loss, metrics)
   2. Preparar los datos (normalización, one-hot)
   3. Entrenar con fit()
   4. Evaluar con evaluate()
   5. Hacer predicciones con predict()
""")

print("\n" + "=" * 70)
print("ISSUE 4(F2) COMPLETADO ✅")
print("=" * 70)
print("\n🎯 El modelo está COMPLETAMENTE construido y listo para compilar")
print("📊 Arquitectura verificada y funcional")
print("🚀 ¡Listo para entrenar!")