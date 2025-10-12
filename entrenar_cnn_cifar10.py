# Issue 2(F3): Entrenamiento del Modelo
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

print("=" * 70)
print("ISSUE 2(F3): ENTRENAMIENTO DEL MODELO CNN")
print("=" * 70)

# ========== CARGAR Y PREPARAR LOS DATOS ==========
print("\n📊 PASO 1: CARGANDO Y PREPARANDO DATOS")
print("=" * 70)

# Cargar CIFAR-10
print("\nCargando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"✅ Dataset cargado:")
print(f"   • x_train shape: {x_train.shape}")
print(f"   • y_train shape: {y_train.shape}")
print(f"   • x_test shape:  {x_test.shape}")
print(f"   • y_test shape:  {y_test.shape}")

# Normalizar las imágenes (0-255 → 0-1)
print("\nNormalizando imágenes...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"✅ Normalización completada:")
print(f"   • Rango de valores: [{x_train.min():.2f}, {x_train.max():.2f}]")
print(f"   • Media: {x_train.mean():.4f}")
print(f"   • Desviación estándar: {x_train.std():.4f}")

# Convertir etiquetas a one-hot encoding
print("\nConvirtiendo etiquetas a one-hot encoding...")
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"✅ Conversión completada:")
print(f"   • y_train shape: {y_train.shape} (one-hot)")
print(f"   • y_test shape:  {y_test.shape} (one-hot)")

# Verificaciones finales de los datos
assert x_train.shape == (50000, 32, 32, 3), "❌ Error en shape de x_train"
assert y_train.shape == (50000, 10), "❌ Error en shape de y_train"
assert 0 <= x_train.min() <= x_train.max() <= 1, "❌ Error en normalización"

print("\n✅ Datos preparados correctamente")

# ========== CONSTRUIR Y COMPILAR EL MODELO ==========
print("\n" + "=" * 70)
print("🏗️  PASO 2: CONSTRUYENDO Y COMPILANDO EL MODELO")
print("=" * 70)

model = Sequential(name="CNN_CIFAR10")

# Arquitectura
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2d_1'))
model.add(MaxPooling2D((2, 2), name='maxpool_1'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv2d_2'))
model.add(MaxPooling2D((2, 2), name='maxpool_2'))
model.add(Flatten(name='flatten'))
model.add(Dense(64, activation='relu', name='dense_hidden'))
model.add(Dense(10, activation='softmax', name='dense_output'))

print("\n✅ Modelo construido")

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo compilado")
print(f"   • Optimizador: Adam")
print(f"   • Loss: categorical_crossentropy")
print(f"   • Métricas: accuracy")

# Mostrar resumen
print("\n" + "-" * 70)
print("RESUMEN DEL MODELO:")
print("-" * 70)
model.summary()

# ========== CONFIGURAR PARÁMETROS DE ENTRENAMIENTO ==========
print("\n" + "=" * 70)
print("⚙️  PASO 3: CONFIGURACIÓN DE ENTRENAMIENTO")
print("=" * 70)

# Parámetros
EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

print(f"""
Parámetros de entrenamiento:
  • Épocas (epochs):              {EPOCHS}
  • Tamaño de batch (batch_size): {BATCH_SIZE}
  • Validación (validation_split): {VALIDATION_SPLIT} ({VALIDATION_SPLIT*100:.0f}%)

Distribución de datos:
  • Datos totales de entrenamiento: {len(x_train):,}
  • Datos para entrenamiento real:  {int(len(x_train) * (1-VALIDATION_SPLIT)):,} ({(1-VALIDATION_SPLIT)*100:.0f}%)
  • Datos para validación:          {int(len(x_train) * VALIDATION_SPLIT):,} ({VALIDATION_SPLIT*100:.0f}%)
  • Datos de test (no se tocan):    {len(x_test):,}

Iteraciones por época:
  • Batches por época: {int(len(x_train) * (1-VALIDATION_SPLIT) / BATCH_SIZE)}
  • Iteraciones totales: {int(len(x_train) * (1-VALIDATION_SPLIT) / BATCH_SIZE) * EPOCHS}
""")

# ========== EXPLICACIÓN DE PARÁMETROS ==========
print("=" * 70)
print("🧠 EXPLICACIÓN DE PARÁMETROS:")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────┐
│ EPOCHS (Épocas)                                                │
├────────────────────────────────────────────────────────────────┤
│ ¿Qué es una época?                                            │
│ • Una pasada COMPLETA por todo el dataset de entrenamiento   │
│ • En cada época, la red ve todas las 45,000 imágenes         │
│ • Los pesos se actualizan múltiples veces por época          │
│                                                                │
│ ¿Por qué 10 épocas?                                           │
│ • Suficiente para ver mejora significativa                    │
│ • No tan largo que cause overfitting excesivo                 │
│ • Balance entre tiempo de entrenamiento y resultados          │
│                                                                │
│ Nota: Modelos profesionales usan 50-200 épocas, pero para    │
│       pruebas iniciales, 10 es razonable.                     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ BATCH_SIZE (Tamaño de lote)                                   │
├────────────────────────────────────────────────────────────────┤
│ ¿Qué es un batch?                                             │
│ • Grupo de imágenes procesadas JUNTAS antes de actualizar    │
│ • Con batch_size=64: procesamos 64 imágenes → actualizamos   │
│                                                                │
│ ¿Por qué 64?                                                  │
│ • Compromiso entre velocidad y estabilidad                    │
│ • Batch pequeño (16): Más actualizaciones, más ruido         │
│ • Batch grande (256): Menos actualizaciones, más estable     │
│ • 64 es un valor estándar que funciona bien                   │
│                                                                │
│ Valores típicos:                                              │
│ • 32: Bueno para GPUs pequeñas                               │
│ • 64: ✅ Valor por defecto recomendado                        │
│ • 128/256: Para GPUs potentes y datasets grandes             │
│                                                                │
│ Regla práctica: Usa potencias de 2 (32, 64, 128...)          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ VALIDATION_SPLIT (División de validación)                     │
├────────────────────────────────────────────────────────────────┤
│ ¿Qué es la validación?                                        │
│ • Subset de datos NO usado para entrenar                      │
│ • Usado para evaluar durante el entrenamiento                │
│ • Permite detectar overfitting                                │
│                                                                │
│ ¿Por qué 0.1 (10%)?                                           │
│ • 10% = 5,000 imágenes para validación                       │
│ • Suficiente para evaluar confiablemente                     │
│ • No resta demasiados datos al entrenamiento                 │
│                                                                │
│ Flujo de datos:                                               │
│   50,000 imágenes totales                                     │
│   ├─ 45,000 (90%) → Entrenamiento (actualiza pesos)         │
│   └─  5,000 (10%) → Validación (solo evalúa)                │
│                                                                │
│ Diferencia con test:                                          │
│ • VALIDACIÓN: Evalúa DURANTE entrenamiento                   │
│ • TEST: Evalúa DESPUÉS del entrenamiento                     │
│ • Test (10,000 imgs) NO se toca hasta el final               │
└────────────────────────────────────────────────────────────────┘
""")

# ========== ENTRENAR EL MODELO ==========
print("\n" + "=" * 70)
print("🚀 PASO 4: ENTRENANDO EL MODELO")
print("=" * 70)

print(f"\nIniciando entrenamiento con {EPOCHS} épocas...")
print("⏱️  Esto puede tomar varios minutos...\n")

# Registrar tiempo de inicio
start_time = time.time()

# ENTRENAR
history = model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1  # Mostrar barra de progreso detallada
)

# Calcular tiempo total
end_time = time.time()
training_time = end_time - start_time

print("\n" + "=" * 70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"⏱️  Tiempo total de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
print(f"⏱️  Tiempo promedio por época: {training_time/EPOCHS:.2f} segundos")

# ========== ANÁLISIS DEL HISTORIAL ==========
print("\n" + "=" * 70)
print("📊 PASO 5: ANÁLISIS DEL HISTORIAL DE ENTRENAMIENTO")
print("=" * 70)

# Extraer métricas finales
final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n📈 MÉTRICAS FINALES (Época {}):\n".format(EPOCHS))
print(f"  ENTRENAMIENTO:")
print(f"    • Loss:     {final_train_loss:.4f}")
print(f"    • Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"\n  VALIDACIÓN:")
print(f"    • Loss:     {final_val_loss:.4f}")
print(f"    • Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

# Detectar overfitting
overfitting_gap = final_train_acc - final_val_acc
print(f"\n  DIFERENCIA (Gap):")
print(f"    • Accuracy gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

if overfitting_gap < 0.05:
    print("    ✅ Bajo overfitting - Modelo generaliza bien")
elif overfitting_gap < 0.10:
    print("    ⚠️  Overfitting moderado - Aceptable")
else:
    print("    ❌ Overfitting alto - Considerar regularización")

# ========== TABLA DE EVOLUCIÓN POR ÉPOCA ==========
print("\n" + "=" * 70)
print("📋 EVOLUCIÓN DETALLADA POR ÉPOCA:")
print("=" * 70)

print("\n┌───────┬─────────────────────┬─────────────────────┐")
print("│ Época │    ENTRENAMIENTO    │     VALIDACIÓN      │")
print("│       │  Loss    │ Accuracy │  Loss    │ Accuracy │")
print("├───────┼──────────┼──────────┼──────────┼──────────┤")

for epoch in range(EPOCHS):
    train_loss = history.history['loss'][epoch]
    train_acc = history.history['accuracy'][epoch]
    val_loss = history.history['val_loss'][epoch]
    val_acc = history.history['val_accuracy'][epoch]
    
    print(f"│  {epoch+1:2d}   │  {train_loss:.4f}  │  {train_acc:.4f}  │  {val_loss:.4f}  │  {val_acc:.4f}  │")

print("└───────┴──────────┴──────────┴──────────┴──────────┘")

# ========== ANÁLISIS DE MEJORA ==========
print("\n" + "=" * 70)
print("📊 ANÁLISIS DE MEJORA:")
print("=" * 70)

# Primera vs última época
first_train_acc = history.history['accuracy'][0]
first_val_acc = history.history['val_accuracy'][0]
train_improvement = (final_train_acc - first_train_acc) * 100
val_improvement = (final_val_acc - first_val_acc) * 100

print(f"""
PROGRESO DESDE LA PRIMERA ÉPOCA:

Entrenamiento:
  • Época 1:  {first_train_acc:.4f} ({first_train_acc*100:.2f}%)
  • Época {EPOCHS}: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
  • Mejora:   +{train_improvement:.2f} puntos porcentuales

Validación:
  • Época 1:  {first_val_acc:.4f} ({first_val_acc*100:.2f}%)
  • Época {EPOCHS}: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
  • Mejora:   +{val_improvement:.2f} puntos porcentuales

La red aprendió exitosamente ✅
""")

# ========== MEJOR ÉPOCA ==========
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_acc = max(history.history['val_accuracy'])

print(f"""
MEJOR RENDIMIENTO EN VALIDACIÓN:
  • Época: {best_epoch}
  • Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
""")

# ========== VERIFICACIONES FINALES ==========
print("=" * 70)
print("✅ VERIFICACIONES FINALES:")
print("=" * 70)

# Verificar que history tiene las claves esperadas
expected_keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
assert all(key in history.history for key in expected_keys), "❌ Faltan métricas en history"
print("✅ History contiene todas las métricas esperadas")

# Verificar que se entrenaron todas las épocas
assert len(history.history['loss']) == EPOCHS, f"❌ Se esperaban {EPOCHS} épocas"
print(f"✅ Se completaron todas las {EPOCHS} épocas")

# Verificar que la accuracy mejoró
assert final_val_acc > first_val_acc, "❌ La validación no mejoró"
print("✅ La accuracy de validación mejoró durante el entrenamiento")

# Verificar que el modelo aprendió algo útil
assert final_val_acc > 0.4, "❌ Accuracy muy baja, modelo no aprendió"
print(f"✅ Accuracy de validación ({final_val_acc*100:.2f}%) es razonable")

# ========== INFORMACIÓN DEL OBJETO HISTORY ==========
print("\n" + "=" * 70)
print("📝 INFORMACIÓN DEL OBJETO HISTORY:")
print("=" * 70)

print(f"""
El objeto 'history' contiene:

history.history: diccionario con las métricas
  • Keys: {list(history.history.keys())}
  
Cada key es una lista con valores por época:
  • len(history.history['loss']): {len(history.history['loss'])} épocas
  
Acceso a datos:
  • history.history['loss'][0]          → Loss de época 1
  • history.history['val_accuracy'][-1] → Val accuracy final
  • history.epoch                        → Lista [0, 1, 2, ..., {EPOCHS-1}]

Este objeto es útil para:
  ✓ Graficar curvas de aprendizaje
  ✓ Detectar overfitting
  ✓ Decidir early stopping
  ✓ Comparar experimentos
""")

# ========== GUARDADO DEL HISTORIAL (OPCIONAL) ==========
print("\n" + "=" * 70)
print("💾 GUARDADO DEL HISTORIAL (Opcional):")
print("=" * 70)

print("""
Para guardar el historial para análisis posterior:

import pickle

# Guardar
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Cargar más tarde
with open('training_history.pkl', 'rb') as f:
    loaded_history = pickle.load(f)
""")

# ========== RESUMEN EJECUTIVO ==========
print("\n" + "=" * 70)
print("📋 RESUMEN EJECUTIVO:")
print("=" * 70)

print(f"""
ENTRENAMIENTO COMPLETADO EXITOSAMENTE ✅

Configuración:
  • Épocas: {EPOCHS}
  • Batch size: {BATCH_SIZE}
  • Validation split: {VALIDATION_SPLIT}
  
Resultados Finales:
  • Train Accuracy: {final_train_acc*100:.2f}%
  • Val Accuracy:   {final_val_acc*100:.2f}%
  • Mejora total:   +{val_improvement:.2f}%
  
Tiempo:
  • Total: {training_time/60:.2f} minutos
  • Por época: {training_time/EPOCHS:.2f} segundos
  
Estado:
  ✅ Modelo entrenado
  ✅ Métricas generadas
  ✅ History guardado en variable
  ✅ Listo para evaluación en test

Próximos pasos:
  1. Evaluar en test set: model.evaluate(x_test, y_test)
  2. Graficar curvas de aprendizaje
  3. Analizar predicciones individuales
  4. Matriz de confusión
""")

print("\n" + "=" * 70)
print("ISSUE 2(F3) COMPLETADO ✅")
print("=" * 70)
print("\n🎉 ¡Entrenamiento exitoso!")
print("📊 El modelo ha aprendido a clasificar imágenes de CIFAR-10")
print("🚀 Listo para evaluación y análisis de resultados")