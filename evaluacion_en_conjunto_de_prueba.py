# Issue 4(F3): Evaluación en Conjunto de Prueba
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

print("=" * 70)
print("ISSUE 4(F3): EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("=" * 70)

# ========== PREPARAR DATOS Y ENTRENAR ==========
print("\n📊 PASO 1: PREPARANDO DATOS Y ENTRENANDO MODELO")
print("=" * 70)

print("\nCargando dataset CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("✅ Dataset cargado")
print(f"   • Training set:   {len(x_train):,} imágenes")
print(f"   • Test set:       {len(x_test):,} imágenes")

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir a one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("✅ Datos normalizados y convertidos a one-hot")

# Construir y compilar modelo
print("\nConstruyendo modelo...")
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
], name="CNN_CIFAR10")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Modelo compilado")

# Entrenar
print("\nEntrenando modelo...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)
print("✅ Entrenamiento completado\n")

# Métricas de validación
val_acc = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Resultados finales de VALIDACIÓN:")
print(f"  • Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  • Loss:     {val_loss:.4f}")

# ========== EVALUACIÓN EN TEST SET ==========
print("\n" + "=" * 70)
print("🧪 PASO 2: EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("=" * 70)

print("\n⚠️  IMPORTANTE:")
print("   El conjunto de test NO se ha usado durante el entrenamiento.")
print("   Representa datos completamente nuevos para el modelo.")
print("   Esta es la evaluación MÁS REALISTA del rendimiento.\n")

print("Evaluando modelo en test set...")
print("(Esto puede tomar unos segundos...)\n")

# EVALUAR EN TEST
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print("\n" + "=" * 70)
print("📊 RESULTADOS EN CONJUNTO DE PRUEBA:")
print("=" * 70)

print(f"""
╔════════════════════════════════════════════════════════════╗
║              RESULTADOS FINALES EN TEST SET                ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Test Loss:     {test_loss:6.4f}                                    ║
║  Test Accuracy: {test_accuracy:6.4f}  ({test_accuracy*100:5.2f}%)                      ║
║                                                            ║
║  Imágenes evaluadas: {len(x_test):,}                              ║
║  Predicciones correctas: {int(test_accuracy * len(x_test)):,}                      ║
║  Predicciones incorrectas: {int((1 - test_accuracy) * len(x_test)):,}                    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")

# ========== COMPARACIÓN VALIDACIÓN VS TEST ==========
print("\n" + "=" * 70)
print("📊 COMPARACIÓN: VALIDACIÓN vs TEST")
print("=" * 70)

# Calcular diferencias
acc_diff = val_acc - test_accuracy
loss_diff = test_loss - val_loss

print(f"""
┌─────────────────┬──────────────┬──────────────┬─────────────┐
│                 │  VALIDACIÓN  │     TEST     │  DIFERENCIA │
├─────────────────┼──────────────┼──────────────┼─────────────┤
│ Accuracy        │   {val_acc:.4f}     │   {test_accuracy:.4f}     │   {acc_diff:+.4f}    │
│ Loss            │   {val_loss:.4f}     │   {test_loss:.4f}     │   {loss_diff:+.4f}    │
└─────────────────┴──────────────┴──────────────┴─────────────┘

Porcentajes:
  • Validación: {val_acc*100:.2f}%
  • Test:       {test_accuracy*100:.2f}%
  • Diferencia: {acc_diff*100:+.2f}%
""")

# ========== INTERPRETACIÓN DE LA DIFERENCIA ==========
print("\n" + "=" * 70)
print("🔍 INTERPRETACIÓN DE LA DIFERENCIA:")
print("=" * 70)

print("\n¿Qué significa esta diferencia?\n")

if abs(acc_diff) < 0.02:
    interpretation = "✅ EXCELENTE"
    explanation = """
    La diferencia es MÍNIMA (< 2%).
    
    Esto significa:
    • El modelo generaliza MUY BIEN
    • No hay overfitting significativo al conjunto de validación
    • El conjunto de validación fue representativo
    • El rendimiento es consistente entre conjuntos
    
    Conclusión: Modelo muy robusto y confiable."""
    
elif abs(acc_diff) < 0.05:
    interpretation = "✅ BUENO"
    explanation = """
    La diferencia es PEQUEÑA (2-5%).
    
    Esto significa:
    • El modelo generaliza BIEN
    • Overfitting mínimo
    • Comportamiento esperado y normal
    • El conjunto de validación fue razonablemente representativo
    
    Conclusión: Modelo confiable con buen equilibrio."""
    
elif abs(acc_diff) < 0.10:
    interpretation = "⚠️  ACEPTABLE"
    explanation = """
    La diferencia es MODERADA (5-10%).
    
    Esto puede indicar:
    • Ligero overfitting al conjunto de validación
    • O variabilidad natural entre subconjuntos
    • El modelo es razonablemente robusto
    
    Conclusión: Modelo funcional pero con margen de mejora."""
    
else:
    interpretation = "🔴 PREOCUPANTE"
    explanation = """
    La diferencia es GRANDE (> 10%).
    
    Esto indica:
    • Overfitting significativo
    • El conjunto de validación no fue representativo
    • O el modelo es inestable
    
    Conclusión: Requiere ajustes (regularización, más datos, etc.)."""

print(f"Estado: {interpretation}")
print(f"Diferencia de accuracy: {acc_diff*100:+.2f}%")
print(explanation)

# ========== ANÁLISIS ADICIONAL: DIRECCIÓN DE LA DIFERENCIA ==========
print("\n" + "-" * 70)
print("ANÁLISIS DE DIRECCIÓN:")
print("-" * 70)

if acc_diff > 0:
    print(f"""
Validación ({val_acc*100:.2f}%) > Test ({test_accuracy*100:.2f}%)

Posibles razones:
1. OVERFITTING: El modelo se ajustó ligeramente al conjunto de validación
   • Aunque no entrenamos directamente con validación, podríamos 
     haber hecho ajustes (implícitos o explícitos) basados en val_accuracy

2. VARIABILIDAD ALEATORIA: Los conjuntos son muestras aleatorias
   • Es normal cierta variación entre subconjuntos
   • Si la diferencia es < 5%, es completamente normal

3. DISTRIBUCIÓN: El test set podría tener ejemplos más difíciles
   • CIFAR-10 tiene imágenes de diferentes dificultades
   • Mala suerte en la partición aleatoria

Evaluación: {'Normal si < 5%' if acc_diff < 0.05 else 'Considerar overfitting'}
""")
else:
    print(f"""
Test ({test_accuracy*100:.2f}%) > Validación ({val_acc*100:.2f}%)

Posibles razones:
1. BUENA SUERTE: El test set tiene ejemplos más fáciles
   • Variación aleatoria favorable

2. VALIDACIÓN MÁS DIFÍCIL: El 10% de validación era más desafiante
   • Puede pasar con conjuntos pequeños

3. MODELO ROBUSTO: Generaliza incluso mejor de lo esperado
   • Señal positiva de buena arquitectura

Evaluación: ✅ Situación favorable, pero inusual
""")

# ========== CONTEXTO PARA CIFAR-10 ==========
print("\n" + "=" * 70)
print("📚 CONTEXTO: RENDIMIENTO TÍPICO EN CIFAR-10")
print("=" * 70)

print(f"""
Benchmarks de accuracy en CIFAR-10:

┌─────────────────────────────┬──────────────┐
│ Tipo de Modelo              │  Accuracy    │
├─────────────────────────────┼──────────────┤
│ Random Guess (baseline)     │    10%       │
│ Shallow MLP                 │    40-50%    │
│ CNN Simple (nuestra)        │  ✓ 60-75%    │ ← Nuestro rango esperado
│ CNN con Data Augmentation   │    75-85%    │
│ ResNet-18                   │    85-90%    │
│ ResNet-50                   │    90-93%    │
│ State-of-the-art (2024)     │    99%+      │
└─────────────────────────────┴──────────────┘

Tu modelo: {test_accuracy*100:.2f}%
""")

# Clasificación del rendimiento
if test_accuracy >= 0.75:
    performance = "🌟 EXCELENTE"
    comment = "Muy por encima del baseline para una CNN simple"
elif test_accuracy >= 0.65:
    performance = "✅ BUENO"
    comment = "Dentro del rango esperado para esta arquitectura"
elif test_accuracy >= 0.55:
    performance = "⚠️  ACEPTABLE"
    comment = "Por debajo del óptimo, hay margen de mejora"
else:
    performance = "🔴 BAJO"
    comment = "Significativamente por debajo de lo esperado"

print(f"\nRendimiento: {performance}")
print(f"Evaluación: {comment}")

# ========== ANÁLISIS DE ERRORES ==========
print("\n" + "=" * 70)
print("🔍 ANÁLISIS DE ERRORES:")
print("=" * 70)

correct_predictions = int(test_accuracy * len(x_test))
incorrect_predictions = len(x_test) - correct_predictions

print(f"""
De {len(x_test):,} imágenes de test:

✅ CORRECTAS:   {correct_predictions:,} ({test_accuracy*100:.2f}%)
❌ INCORRECTAS: {incorrect_predictions:,} ({(1-test_accuracy)*100:.2f}%)

Tasa de error: {(1-test_accuracy)*100:.2f}%

Esto significa:
  • De cada 100 imágenes, el modelo clasifica correctamente ~{int(test_accuracy*100)}
  • De cada 100 imágenes, el modelo se equivoca en ~{int((1-test_accuracy)*100)}
""")

# ========== POSIBLES MEJORAS ==========
print("\n" + "=" * 70)
print("💡 POSIBLES MEJORAS AL MODELO:")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────┐
│ 1️⃣  ARQUITECTURA                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ✓ Añadir más capas convolucionales (3-4 bloques total)       │
│     model.add(Conv2D(128, (3,3), activation='relu'))         │
│     model.add(MaxPooling2D((2,2)))                           │
│                                                                │
│ ✓ Aumentar número de filtros progresivamente                  │
│     32 → 64 → 128 → 256                                       │
│                                                                │
│ ✓ Usar Batch Normalization                                    │
│     model.add(BatchNormalization())                           │
│     Acelera entrenamiento y mejora estabilidad                │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 2️⃣  REGULARIZACIÓN                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ✓ Añadir Dropout para reducir overfitting                     │
│     model.add(Dropout(0.3))                                   │
│     model.add(Dropout(0.5))  # Antes de última capa          │
│                                                                │
│ ✓ Usar regularización L2                                      │
│     Dense(64, kernel_regularizer=l2(0.001))                   │
│                                                                │
│ ✓ Early Stopping                                              │
│     EarlyStopping(patience=5, restore_best_weights=True)      │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 3️⃣  DATA AUGMENTATION                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ✓ Aumentar artificialmente el dataset                         │
│     datagen = ImageDataGenerator(                             │
│         rotation_range=15,                                    │
│         width_shift_range=0.1,                                │
│         height_shift_range=0.1,                               │
│         horizontal_flip=True                                  │
│     )                                                          │
│                                                                │
│ Impacto esperado: +5-10% accuracy                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 4️⃣  HIPERPARÁMETROS                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ✓ Ajustar learning rate                                       │
│     optimizer=Adam(learning_rate=0.0001)  # Más conservador  │
│                                                                │
│ ✓ Usar learning rate scheduler                                │
│     ReduceLROnPlateau(factor=0.5, patience=3)                 │
│                                                                │
│ ✓ Entrenar más épocas (20-50)                                 │
│                                                                │
│ ✓ Probar diferentes batch sizes (32, 128)                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 5️⃣  TÉCNICAS AVANZADAS                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ✓ Transfer Learning (usar modelo pre-entrenado)               │
│     VGG16, ResNet50, EfficientNet                             │
│                                                                │
│ ✓ Ensemble de modelos                                         │
│     Combinar predicciones de varios modelos                   │
│                                                                │
│ ✓ Test-Time Augmentation (TTA)                                │
│     Predecir sobre versiones aumentadas y promediar           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
""")

# ========== PRIORIZACIÓN DE MEJORAS ==========
print("\n" + "=" * 70)
print("🎯 PRIORIZACIÓN DE MEJORAS:")
print("=" * 70)

if test_accuracy < 0.60:
    priority = """
    PRIORIDAD ALTA (Modelo necesita mejoras sustanciales):
    
    1. 🔴 CRÍTICO: Revisar arquitectura
       • Añadir más capas convolucionales
       • Aumentar número de filtros
       
    2. 🔴 CRÍTICO: Verificar datos
       • Asegurar normalización correcta
       • Verificar one-hot encoding
       
    3. 🟡 IMPORTANTE: Entrenar más tiempo
       • Aumentar a 20-30 épocas
       • Monitorear curvas de aprendizaje
    """
elif test_accuracy < 0.70:
    priority = """
    PRIORIDAD MEDIA (Modelo funciona pero tiene margen):
    
    1. 🟡 RECOMENDADO: Data Augmentation
       • Impacto: +5-10% accuracy
       • Implementación simple
       
    2. 🟡 RECOMENDADO: Regularización
       • Añadir Dropout(0.3, 0.5)
       • Ayuda con overfitting
       
    3. 🟢 OPCIONAL: Más capas
       • Un tercer bloque convolucional
       • Conv2D(128) + MaxPooling
    """
else:
    priority = """
    PRIORIDAD BAJA (Modelo funciona bien):
    
    1. 🟢 OPCIONAL: Fine-tuning
       • Ajustar learning rate
       • Experimentar con batch size
       
    2. 🟢 OPCIONAL: Técnicas avanzadas
       • Batch Normalization
       • Learning rate scheduling
       
    3. 🔵 EXPLORACIÓN: Transfer Learning
       • Solo si necesitas > 85% accuracy
    """

print(priority)

# ========== CONCLUSIONES FINALES ==========
print("\n" + "=" * 70)
print("📋 CONCLUSIONES FINALES:")
print("=" * 70)

# Generar conclusión personalizada
if test_accuracy >= 0.70:
    conclusion_quality = "✅ EXITOSO"
    conclusion_text = f"""
El modelo ha alcanzado una accuracy de {test_accuracy*100:.2f}% en el test set,
lo cual es un resultado BUENO para una CNN simple en CIFAR-10.

Logros:
• Superó el baseline de CNNs simples (~60%)
• Generaliza bien a datos nuevos
• Diferencia validación-test aceptable: {abs(acc_diff)*100:.2f}%

El modelo está listo para uso en aplicaciones reales, aunque
mejoras adicionales podrían aumentar su rendimiento.
    """
elif test_accuracy >= 0.60:
    conclusion_quality = "✅ ACEPTABLE"
    conclusion_text = f"""
El modelo ha alcanzado una accuracy de {test_accuracy*100:.2f}% en el test set,
lo cual está en el rango esperado para una CNN simple en CIFAR-10.

Logros:
• Supera ampliamente el random guess (10%)
• Está en el rango típico (60-75%)
• Demuestra que la arquitectura CNN funciona

Con las mejoras sugeridas (Data Augmentation, Dropout, más capas),
se podría alcanzar fácilmente 75-80% de accuracy.
    """
else:
    conclusion_quality = "⚠️  MEJORABLE"
    conclusion_text = f"""
El modelo ha alcanzado una accuracy de {test_accuracy*100:.2f}% en el test set,
lo cual está por debajo del rendimiento típico para CNNs en CIFAR-10.

Observaciones:
• Hay margen significativo de mejora
• La arquitectura necesita refinamiento
• Considerar aumentar capacidad del modelo

Recomendación: Implementar las mejoras prioritarias mencionadas,
especialmente añadir más capas convolucionales y usar Data Augmentation.
    """

print(f"\n{conclusion_quality}\n")
print(conclusion_text)

# ========== RESUMEN EJECUTIVO FINAL ==========
print("\n" + "=" * 70)
print("📊 RESUMEN EJECUTIVO:")
print("=" * 70)

print(f"""
╔════════════════════════════════════════════════════════════╗
║                   EVALUACIÓN FINAL                         ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Test Accuracy:      {test_accuracy*100:5.2f}%                             ║
║  Test Loss:          {test_loss:6.4f}                                ║
║                                                            ║
║  Validación Accuracy: {val_acc*100:5.2f}%                            ║
║  Diferencia Val-Test: {acc_diff*100:+5.2f}%                            ║
║                                                            ║
║  Imágenes correctas:  {correct_predictions:,} / {len(x_test):,}                   ║
║  Tasa de error:       {(1-test_accuracy)*100:5.2f}%                             ║
║                                                            ║
║  Rendimiento:         {performance:20s}            ║
║  Generalización:      {interpretation:20s}            ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

ESTADO: ✅ Evaluación completada exitosamente

PRÓXIMOS PASOS SUGERIDOS:
  1. Analizar matriz de confusión (¿qué clases confunde?)
  2. Visualizar predicciones incorrectas
  3. Implementar mejoras prioritarias
  4. Re-entrenar y comparar resultados
""")

print("\n" + "=" * 70)
print("ISSUE 4(F3) COMPLETADO ✅")
print("=" * 70)
print("\n🎯 Evaluación en test set completada")
print("📊 Capacidad de generalización medida")
print("💡 Recomendaciones de mejora proporcionadas")
print("✅ Modelo listo para análisis detallado o despliegue")