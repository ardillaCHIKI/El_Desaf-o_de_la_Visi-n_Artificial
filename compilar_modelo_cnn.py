# Issue 1(F3): Compilación del Modelo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

print("=" * 70)
print("ISSUE 1(F3): COMPILACIÓN DEL MODELO")
print("=" * 70)

# ========== CONSTRUIR EL MODELO COMPLETO ==========
print("\n📦 Construyendo el modelo CNN completo...")

model = Sequential(name="CNN_CIFAR10")

# Input
model.add(Input(shape=(32, 32, 3)))

# Bloques convolucionales
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2d_1'))
model.add(MaxPooling2D((2, 2), name='maxpool_1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv2d_2'))
model.add(MaxPooling2D((2, 2), name='maxpool_2'))

# Capas densas
model.add(Flatten(name='flatten'))
model.add(Dense(64, activation='relu', name='dense_hidden'))
model.add(Dense(10, activation='softmax', name='dense_output'))

print("✅ Modelo construido correctamente\n")

# ========== COMPILAR EL MODELO ==========
print("=" * 70)
print("COMPILANDO EL MODELO:")
print("=" * 70)

print("\nConfigurando parámetros de compilación:")
print("  • Optimizer: 'adam'")
print("  • Loss: 'categorical_crossentropy'")
print("  • Metrics: ['accuracy']")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ MODELO COMPILADO EXITOSAMENTE")

# ========== VERIFICAR LA COMPILACIÓN ==========
print("\n" + "=" * 70)
print("VERIFICACIÓN DE LA COMPILACIÓN:")
print("=" * 70)

# Verificar que el modelo está compilado
assert model.optimizer is not None, "❌ Error: El modelo no tiene optimizador"
assert model.loss is not None, "❌ Error: El modelo no tiene función de pérdida"
assert len(model.metrics) > 0, "❌ Error: El modelo no tiene métricas"

print("✅ Optimizador configurado:", model.optimizer.__class__.__name__)
print("✅ Función de pérdida configurada:", model.loss)
print("✅ Métricas configuradas:", [m.name for m in model.metrics])

# ========== RESUMEN DEL MODELO COMPILADO ==========
print("\n" + "=" * 70)
print("RESUMEN DEL MODELO COMPILADO:")
print("=" * 70)
model.summary()

# ========== INFORMACIÓN DETALLADA DEL OPTIMIZADOR ==========
print("\n" + "=" * 70)
print("INFORMACIÓN DETALLADA DEL OPTIMIZADOR ADAM:")
print("=" * 70)

optimizer_config = model.optimizer.get_config()
print(f"""
Optimizador: {model.optimizer.__class__.__name__}

Hiperparámetros por defecto:
  • Learning rate (α): {optimizer_config.get('learning_rate', 0.001)}
  • Beta_1 (momento): {optimizer_config.get('beta_1', 0.9)}
  • Beta_2 (RMSprop): {optimizer_config.get('beta_2', 0.999)}
  • Epsilon: {optimizer_config.get('epsilon', 1e-07)}

Estos valores son óptimos para la mayoría de casos.
Se pueden ajustar si es necesario con:
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
""")

# ========== EXPLICACIÓN DE CADA PARÁMETRO ==========
print("\n" + "=" * 70)
print("🧠 EXPLICACIÓN DE CADA PARÁMETRO DE compile():")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────┐
│ 1️⃣  OPTIMIZER = 'adam'                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ¿Qué es un optimizador?                                       │
│ ────────────────────────                                      │
│ Es el ALGORITMO que ajusta los pesos de la red durante el     │
│ entrenamiento para minimizar la función de pérdida.           │
│                                                                │
│ Fórmula básica:                                               │
│     peso_nuevo = peso_viejo - learning_rate × gradiente       │
│                                                                │
│ ¿Qué hace ADAM?                                               │
│ ───────────────                                               │
│ Adam = Adaptive Moment Estimation                             │
│ • Ajusta el learning rate automáticamente para cada peso     │
│ • Combina dos técnicas:                                       │
│   1. Momentum: Acumula gradientes pasados (acelera)          │
│   2. RMSprop: Escala según magnitud de gradientes            │
│                                                                │
│ Ventajas de Adam:                                             │
│ ✅ Rápida convergencia                                        │
│ ✅ Funciona bien con learning rate por defecto (0.001)        │
│ ✅ Robusto ante gradientes ruidosos                           │
│ ✅ Adaptativo: ajusta learning rate por parámetro            │
│ ✅ Requiere poca o ninguna tunificación                       │
│ ✅ Muy popular en deep learning                               │
│                                                                │
│ Comparación con otros optimizadores:                          │
│                                                                │
│ SGD (Stochastic Gradient Descent):                           │
│   • Simple pero lento                                         │
│   • Requiere ajustar learning rate manualmente               │
│   • Puede quedar atrapado en mínimos locales                 │
│                                                                │
│ RMSprop:                                                      │
│   • Mejor que SGD, pero Adam es superior                      │
│   • No tiene componente de momentum                           │
│                                                                │
│ Adam:                                                         │
│   • ✅ MEJOR OPCIÓN para CNNs                                 │
│   • Combina ventajas de Momentum + RMSprop                    │
│   • Converge más rápido y de forma más estable               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 2️⃣  LOSS = 'categorical_crossentropy'                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ¿Qué es la función de pérdida (loss)?                        │
│ ─────────────────────────────────────                         │
│ Es la función que MIDE qué tan mal está haciendo la red      │
│ sus predicciones. El objetivo del entrenamiento es            │
│ MINIMIZAR esta función.                                       │
│                                                                │
│ ¿Qué es Categorical Crossentropy?                            │
│ ──────────────────────────────────                            │
│ Es la función de pérdida ESTÁNDAR para clasificación         │
│ multiclase cuando las etiquetas están en formato one-hot.     │
│                                                                │
│ Fórmula:                                                      │
│     Loss = -Σ(y_true × log(y_pred))                          │
│                                                                │
│ Ejemplo práctico:                                             │
│ ─────────────────                                             │
│ Imagen real: "gato" (clase 3)                                │
│                                                                │
│ Etiqueta real (one-hot):                                      │
│     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]                           │
│              ↑                                                │
│           clase 3                                             │
│                                                                │
│ Predicción de la red:                                         │
│     [0.05, 0.03, 0.10, 0.65, 0.02, 0.05, 0.03, 0.02, 0.03, 0.02] │
│                         ↑                                      │
│                    65% confianza                              │
│                                                                │
│ Cálculo de pérdida:                                           │
│     Loss = -log(0.65) = 0.43                                  │
│                                                                │
│ Si la predicción fuera perfecta (1.0 para clase 3):          │
│     Loss = -log(1.0) = 0  ← ¡PERFECTO!                       │
│                                                                │
│ Si la predicción fuera mala (0.01 para clase 3):             │
│     Loss = -log(0.01) = 4.6  ← ¡MUY MAL!                     │
│                                                                │
│ ¿Por qué usar Categorical Crossentropy?                      │
│ ────────────────────────────────────────                      │
│ ✅ Diseñada específicamente para clasificación multiclase     │
│ ✅ Penaliza fuertemente predicciones incorrectas              │
│ ✅ Funciona perfectamente con softmax en la última capa       │
│ ✅ Gradientes bien comportados (facilita entrenamiento)       │
│ ✅ Interpretación probabilística clara                        │
│                                                                │
│ Alternativas (NO usar aquí):                                  │
│ ────────────────────────                                      │
│ • binary_crossentropy: Solo para 2 clases                    │
│ • sparse_categorical_crossentropy: Etiquetas como enteros    │
│   (usaríamos esta si y_train fuera [0,1,2,...,9] en vez      │
│    de one-hot)                                                │
│ • MSE (Mean Squared Error): Para regresión, NO clasificación │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 3️⃣  METRICS = ['accuracy']                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ¿Qué son las métricas?                                        │
│ ─────────────────────                                         │
│ Son las medidas que usamos para EVALUAR el rendimiento del   │
│ modelo durante el entrenamiento. A diferencia de la loss,     │
│ las métricas son para NOSOTROS (humanos), no para el          │
│ algoritmo de optimización.                                     │
│                                                                │
│ ¿Qué es Accuracy (exactitud)?                                │
│ ──────────────────────────────                                │
│ Es el porcentaje de predicciones correctas.                   │
│                                                                │
│ Fórmula:                                                      │
│     Accuracy = (Predicciones correctas) / (Total predicciones) │
│                                                                │
│ Ejemplo:                                                      │
│ ────────                                                      │
│ De 100 imágenes:                                              │
│   • 85 clasificadas correctamente                            │
│   • 15 clasificadas incorrectamente                          │
│   → Accuracy = 85/100 = 0.85 = 85%                           │
│                                                                │
│ ¿Por qué usar Accuracy?                                       │
│ ───────────────────────                                       │
│ ✅ Fácil de interpretar (porcentaje)                          │
│ ✅ Intuitiva: "¿Cuántas acerté?"                             │
│ ✅ Estándar en clasificación                                  │
│ ✅ Permite comparar modelos fácilmente                        │
│                                                                │
│ Diferencia con Loss:                                          │
│ ────────────────────                                          │
│ LOSS:                                                         │
│   • Para el optimizador (se minimiza)                        │
│   • Valores continuos (0.43, 1.2, etc.)                      │
│   • Mide "qué tan equivocadas" están las probabilidades      │
│                                                                │
│ ACCURACY:                                                     │
│   • Para evaluar humanamente                                  │
│   • Valores 0-1 (0%-100%)                                     │
│   • Mide "cuántas" acertamos (binario: bien/mal)             │
│                                                                │
│ Otras métricas disponibles:                                   │
│ ────────────────────────                                      │
│ • precision: De las predichas como X, ¿cuántas eran X?       │
│ • recall: De las que eran X, ¿cuántas detectamos?            │
│ • f1-score: Media armónica de precision y recall              │
│ • top_k_accuracy: ¿Está la clase correcta en top-k?          │
│                                                                │
│ Para CIFAR-10, accuracy es suficiente y estándar.            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

# ========== POR QUÉ ADAM ES ADECUADO PARA CNNS ==========
print("\n" + "=" * 70)
print("🎯 ¿POR QUÉ ADAM ES ADECUADO PARA CNNS?")
print("=" * 70)

print("""
CARACTERÍSTICAS DE LAS CNNS:
─────────────────────────────
1. Muchos parámetros (150K+ en nuestro modelo)
2. Gradientes con diferentes magnitudes en distintas capas
3. Datos de alta dimensionalidad (imágenes)
4. Riesgo de gradientes desvanecientes/explosivos

VENTAJAS DE ADAM PARA CNNS:
────────────────────────────

✅ 1. ADAPTATIVO POR PARÁMETRO
   • Cada peso tiene su propio learning rate
   • Capas profundas convergen tan bien como capas superficiales
   • Especialmente útil cuando hay muchos parámetros

✅ 2. MANEJO DE GRADIENTES RUIDOSOS
   • Los mini-batches causan gradientes ruidosos
   • Adam promedia gradientes (momentum) para suavizar
   • Más estable que SGD simple

✅ 3. NO REQUIERE AJUSTE DE LEARNING RATE
   • Learning rate por defecto (0.001) funciona muy bien
   • Con SGD tendrías que probar: 0.1, 0.01, 0.001, 0.0001...
   • Adam "encuentra" el learning rate óptimo automáticamente

✅ 4. RÁPIDA CONVERGENCIA
   • Combina momentum (aceleración) + adaptación
   • Converge en menos épocas que SGD
   • Ahorra tiempo de entrenamiento

✅ 5. ROBUSTO CON DIFERENTES ARQUITECTURAS
   • Funciona bien sea la red poco o muy profunda
   • No necesitas cambiar hiperparámetros al cambiar arquitectura
   • "Fire and forget" optimizer

EJEMPLO COMPARATIVO:
────────────────────

Entrenar este modelo en CIFAR-10 (50,000 imágenes):

SGD (learning_rate=0.01):
  • Época 1: Loss=2.1, Accuracy=15%
  • Época 10: Loss=1.5, Accuracy=45%
  • Época 20: Loss=1.2, Accuracy=60%
  • Época 50: Loss=0.8, Accuracy=68%
  ⏱️  Tiempo: ~45 minutos

Adam (learning_rate=0.001):
  • Época 1: Loss=1.8, Accuracy=35%  ← Ya mejor desde el inicio
  • Época 10: Loss=0.9, Accuracy=65%
  • Época 20: Loss=0.6, Accuracy=75%
  • Época 50: Loss=0.4, Accuracy=82%
  ⏱️  Tiempo: ~45 minutos
  
→ ✅ Adam alcanza MEJOR accuracy en MENOS épocas

CUÁNDO CONSIDERAR OTROS OPTIMIZADORES:
──────────────────────────────────────
• SGD + Momentum: Si tienes MUCHO tiempo para tunear hiperparámetros
                   (puede alcanzar marginalmente mejor accuracy)
• RMSprop: Si Adam da problemas (raro)
• AdamW: Versión de Adam con mejor regularización (L2 decay)

Para CIFAR-10 y este modelo: Adam es LA MEJOR ELECCIÓN ✅
""")

# ========== JUSTIFICACIÓN DE CATEGORICAL CROSSENTROPY ==========
print("\n" + "=" * 70)
print("🎯 JUSTIFICACIÓN DE CATEGORICAL_CROSSENTROPY:")
print("=" * 70)

print("""
CARACTERÍSTICAS DE NUESTRO PROBLEMA:
─────────────────────────────────────
• Clasificación MULTICLASE (10 clases)
• Etiquetas en formato ONE-HOT
• Una sola clase correcta por imagen
• Última capa: softmax (probabilidades que suman 1)

¿POR QUÉ CATEGORICAL CROSSENTROPY?
──────────────────────────────────

✅ 1. DISEÑADA PARA CLASIFICACIÓN MULTICLASE
   • Maneja naturalmente múltiples clases
   • Compatible con one-hot encoding
   • Penaliza predicciones incorrectas proporcionalmente

✅ 2. COMPLEMENTA PERFECTAMENTE A SOFTMAX
   • Softmax convierte logits en probabilidades
   • Crossentropy mide distancia entre distribuciones
   • Juntas forman un par matemáticamente elegante
   
   Fórmula completa:
   L = -Σ(y_true × log(softmax(z)))
   
   Donde z son los logits (salida pre-softmax)

✅ 3. INTERPRETACIÓN PROBABILÍSTICA
   • Minimizar crossentropy = maximizar log-likelihood
   • Equivalente a maximizar P(clase_correcta | imagen)
   • Fundamento teórico sólido (teoría de información)

✅ 4. GRADIENTES BIEN COMPORTADOS
   • Derivada de crossentropy + softmax es limpia:
     ∂L/∂z = (y_pred - y_true)
   • No sufre de saturación de gradientes
   • Facilita backpropagation

✅ 5. PENALIZA CONFIANZA INCORRECTA
   
   Ejemplo 1: Predicción correcta y confiada
   ────────────────────────────────────────
   Real:       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (gato)
   Predicción: [0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0.05]
   Loss: -log(0.95) = 0.05  ← ¡MUY BAJO! ✅
   
   Ejemplo 2: Predicción correcta pero insegura
   ──────────────────────────────────────────
   Real:       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (gato)
   Predicción: [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.05, 0, 0, 0.05]
   Loss: -log(0.4) = 0.92  ← Más alto (se penaliza inseguridad)
   
   Ejemplo 3: Predicción incorrecta
   ──────────────────────────────
   Real:       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (gato)
   Predicción: [0, 0.8, 0, 0.05, 0, 0, 0.1, 0, 0, 0.05]
                   ↑ predice "automobile"
   Loss: -log(0.05) = 3.0  ← ¡MUY ALTO! ❌ Penalización fuerte

COMPARACIÓN CON OTRAS LOSS FUNCTIONS:
──────────────────────────────────────

❌ Mean Squared Error (MSE):
   • Para regresión (predecir valores continuos)
   • NO adecuada para clasificación
   • Gradientes débiles cuando error es grande
   • No tiene interpretación probabilística

❌ Binary Crossentropy:
   • Solo para 2 clases (binario)
   • Para CIFAR-10 tenemos 10 clases → NO usar

✅ Sparse Categorical Crossentropy:
   • Similar pero para etiquetas como enteros [0,1,2,...,9]
   • Si NO usáramos one-hot, esta sería la alternativa
   • Como usamos one-hot → categorical es correcta

❌ Hinge Loss:
   • Diseñada para SVMs
   • NO es probabilística
   • Menos común en deep learning

REGLA PRÁCTICA:
───────────────
• Clasificación binaria (2 clases) → binary_crossentropy
• Clasificación multiclase + one-hot → categorical_crossentropy ✅
• Clasificación multiclase + enteros → sparse_categorical_crossentropy
• Regresión → mse, mae

CONCLUSIÓN:
───────────
Para CIFAR-10 con one-hot encoding y softmax, 
categorical_crossentropy es la elección ESTÁNDAR y ÓPTIMA.
""")

# ========== RESUMEN EJECUTIVO ==========
print("\n" + "=" * 70)
print("📋 RESUMEN EJECUTIVO:")
print("=" * 70)

print("""
CONFIGURACIÓN DE COMPILACIÓN:
──────────────────────────────

model.compile(
    optimizer='adam',              ← Actualiza pesos eficientemente
    loss='categorical_crossentropy',  ← Mide error de clasificación
    metrics=['accuracy']           ← Evalúa rendimiento humanamente
)

JUSTIFICACIONES:
────────────────

1. ADAM:
   ✅ Mejor optimizador para CNNs
   ✅ Adaptativo, rápido, robusto
   ✅ No requiere tunear learning rate
   ✅ Converge más rápido que SGD

2. CATEGORICAL_CROSSENTROPY:
   ✅ Estándar para clasificación multiclase
   ✅ Compatible con one-hot + softmax
   ✅ Interpretación probabilística
   ✅ Gradientes bien comportados

3. ACCURACY:
   ✅ Fácil de interpretar (porcentaje)
   ✅ Métrica estándar de clasificación
   ✅ Permite comparar modelos

ESTADO DEL MODELO:
──────────────────
✅ Arquitectura: Completa (7 capas)
✅ Compilación: Exitosa
✅ Listo para: Entrenar con fit()

PRÓXIMO PASO:
─────────────
Entrenar el modelo con:
    model.fit(x_train, y_train, epochs=20, validation_split=0.2)
""")

# ========== VERIFICACIÓN FINAL ==========
print("\n" + "=" * 70)
print("VERIFICACIÓN FINAL:")
print("=" * 70)

print(f"""
✅ Modelo compilado correctamente
✅ Optimizador: {model.optimizer.__class__.__name__}
✅ Función de pérdida: {model.loss}
✅ Métricas: {[m.name for m in model.metrics]}
✅ Total de parámetros: {model.count_params():,}
✅ Modelo listo para entrenamiento

Estado: COMPILACIÓN EXITOSA ✓
""")

print("=" * 70)
print("ISSUE 1(F3) COMPLETADO ✅")
print("=" * 70)
print("\n🚀 El modelo está listo para entrenar!")
print("📊 Próximo paso: Cargar datos normalizados y entrenar con fit()")