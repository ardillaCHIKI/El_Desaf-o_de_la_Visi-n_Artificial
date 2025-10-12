# Issue 3(F3): Visualización de Curvas de Aprendizaje
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("ISSUE 3(F3): VISUALIZACIÓN DE CURVAS DE APRENDIZAJE")
print("=" * 70)

# ========== PREPARAR DATOS Y ENTRENAR (Resumen del issue anterior) ==========
print("\n📊 PASO 1: PREPARANDO DATOS Y ENTRENANDO MODELO")
print("=" * 70)

print("\nCargando y preparando CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("✅ Datos preparados")

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

print("\nEntrenando modelo (esto puede tomar varios minutos)...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=0  # Silencioso para no saturar la salida
)
print("✅ Entrenamiento completado\n")

# ========== EXTRAER DATOS DEL HISTORIAL ==========
print("=" * 70)
print("📈 PASO 2: EXTRAYENDO DATOS DEL HISTORIAL")
print("=" * 70)

# Extraer métricas
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_acc) + 1)

print(f"\nDatos extraídos correctamente:")
print(f"  • Número de épocas: {len(train_acc)}")
print(f"  • Métricas disponibles: {list(history.history.keys())}")
print(f"\nÚltima época:")
print(f"  • Train Accuracy: {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
print(f"  • Val Accuracy:   {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")
print(f"  • Train Loss:     {train_loss[-1]:.4f}")
print(f"  • Val Loss:       {val_loss[-1]:.4f}")

# ========== CREAR VISUALIZACIONES ==========
print("\n" + "=" * 70)
print("📊 PASO 3: GENERANDO VISUALIZACIONES")
print("=" * 70)

# Configurar estilo general
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Curvas de Aprendizaje - CNN en CIFAR-10', fontsize=16, fontweight='bold')

# ========== GRÁFICO 1: PRECISIÓN (ACCURACY) ==========
print("\nGenerando gráfico de Precisión...")

ax1.plot(epochs_range, train_acc, 'b-o', label='Entrenamiento', linewidth=2, markersize=6)
ax1.plot(epochs_range, val_acc, 'r-s', label='Validación', linewidth=2, markersize=6)

ax1.set_xlabel('Época', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precisión (Accuracy)', fontsize=12, fontweight='bold')
ax1.set_title('Precisión: Entrenamiento vs Validación', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(1, len(train_acc))
ax1.set_ylim(0, 1)

# Añadir anotaciones en puntos clave
max_val_acc_idx = np.argmax(val_acc)
ax1.annotate(f'Mejor: {val_acc[max_val_acc_idx]:.3f}',
             xy=(max_val_acc_idx + 1, val_acc[max_val_acc_idx]),
             xytext=(10, -15), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

print("✅ Gráfico de Precisión generado")

# ========== GRÁFICO 2: PÉRDIDA (LOSS) ==========
print("Generando gráfico de Pérdida...")

ax2.plot(epochs_range, train_loss, 'b-o', label='Entrenamiento', linewidth=2, markersize=6)
ax2.plot(epochs_range, val_loss, 'r-s', label='Validación', linewidth=2, markersize=6)

ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
ax2.set_ylabel('Pérdida (Loss)', fontsize=12, fontweight='bold')
ax2.set_title('Pérdida: Entrenamiento vs Validación', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, len(train_loss))

# Añadir anotaciones en puntos clave
min_val_loss_idx = np.argmin(val_loss)
ax2.annotate(f'Mínimo: {val_loss[min_val_loss_idx]:.3f}',
             xy=(min_val_loss_idx + 1, val_loss[min_val_loss_idx]),
             xytext=(10, 15), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

print("✅ Gráfico de Pérdida generado")

plt.tight_layout()
plt.savefig('curvas_aprendizaje.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráficos guardados como 'curvas_aprendizaje.png'")
plt.show()

# ========== ANÁLISIS AUTOMÁTICO DEL COMPORTAMIENTO ==========
print("\n" + "=" * 70)
print("🔍 PASO 4: ANÁLISIS AUTOMÁTICO DEL COMPORTAMIENTO")
print("=" * 70)

# Calcular diferencias y tendencias
final_gap = train_acc[-1] - val_acc[-1]
loss_gap = val_loss[-1] - train_loss[-1]
val_acc_trend = val_acc[-1] - val_acc[-3] if len(val_acc) >= 3 else 0
val_loss_trend = val_loss[-1] - val_loss[-3] if len(val_loss) >= 3 else 0

print("\n📊 MÉTRICAS CLAVE:\n")
print(f"Accuracy final:")
print(f"  • Entrenamiento: {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
print(f"  • Validación:    {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")
print(f"  • Diferencia:    {final_gap:.4f} ({final_gap*100:.2f}%)")

print(f"\nLoss final:")
print(f"  • Entrenamiento: {train_loss[-1]:.4f}")
print(f"  • Validación:    {val_loss[-1]:.4f}")
print(f"  • Diferencia:    {loss_gap:.4f}")

print(f"\nTendencia (últimas 3 épocas):")
print(f"  • Val Accuracy: {'+' if val_acc_trend > 0 else ''}{val_acc_trend:.4f} {'↑' if val_acc_trend > 0 else '↓'}")
print(f"  • Val Loss:     {'+' if val_loss_trend > 0 else ''}{val_loss_trend:.4f} {'↑' if val_loss_trend > 0 else '↓'}")

# ========== DIAGNÓSTICO DETALLADO ==========
print("\n" + "=" * 70)
print("🩺 DIAGNÓSTICO DETALLADO:")
print("=" * 70)

# Clasificación del comportamiento
print("\n" + "─" * 70)
print("ANÁLISIS DE OVERFITTING/UNDERFITTING:")
print("─" * 70)

# 1. Análisis de overfitting
print("\n1️⃣  OVERFITTING (Sobreajuste):")
if final_gap > 0.15:
    overfitting_level = "🔴 ALTO"
    overfitting_desc = "El modelo memoriza datos de entrenamiento"
elif final_gap > 0.08:
    overfitting_level = "🟡 MODERADO"
    overfitting_desc = "Hay sobreajuste, pero controlable"
elif final_gap > 0.03:
    overfitting_level = "🟢 BAJO"
    overfitting_desc = "Overfitting mínimo, buen equilibrio"
else:
    overfitting_level = "✅ NINGUNO"
    overfitting_desc = "Excelente generalización"

print(f"   Nivel: {overfitting_level}")
print(f"   Gap de accuracy: {final_gap:.4f} ({final_gap*100:.2f}%)")
print(f"   Interpretación: {overfitting_desc}")

# 2. Análisis de underfitting
print("\n2️⃣  UNDERFITTING (Subajuste):")
if train_acc[-1] < 0.5:
    underfitting_level = "🔴 ALTO"
    underfitting_desc = "El modelo no aprende bien los datos"
elif train_acc[-1] < 0.7:
    underfitting_level = "🟡 MODERADO"
    underfitting_desc = "El modelo podría aprender más"
else:
    underfitting_level = "✅ NINGUNO"
    underfitting_desc = "El modelo aprende adecuadamente"

print(f"   Nivel: {underfitting_level}")
print(f"   Train accuracy: {train_acc[-1]:.4f} ({train_acc[-1]*100:.2f}%)")
print(f"   Interpretación: {underfitting_desc}")

# 3. Análisis de convergencia
print("\n3️⃣  CONVERGENCIA:")
if abs(val_acc_trend) < 0.005 and abs(val_loss_trend) < 0.01:
    convergence = "✅ CONVERGIDA"
    convergence_desc = "Las métricas se han estabilizado"
elif val_acc_trend > 0 and val_loss_trend < 0:
    convergence = "🟢 MEJORANDO"
    convergence_desc = "El modelo sigue aprendiendo"
elif val_acc_trend < 0 and val_loss_trend > 0:
    convergence = "🔴 EMPEORANDO"
    convergence_desc = "Posible overfitting progresivo"
else:
    convergence = "🟡 INESTABLE"
    convergence_desc = "Las métricas fluctúan"

print(f"   Estado: {convergence}")
print(f"   Interpretación: {convergence_desc}")

# ========== INTERPRETACIÓN DETALLADA ==========
print("\n" + "=" * 70)
print("📝 INTERPRETACIÓN DETALLADA DE LAS CURVAS:")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────┐
│ INTERPRETACIÓN DE LAS CURVAS DE APRENDIZAJE                   │
└────────────────────────────────────────────────────────────────┘

🔵 CURVA DE ACCURACY (Precisión):
──────────────────────────────────
""")

# Análisis específico de accuracy
if train_acc[-1] > val_acc[-1]:
    print("""✓ La curva de ENTRENAMIENTO está por ENCIMA de VALIDACIÓN
  → Esto es NORMAL y esperado
  → El modelo ve los datos de entrenamiento durante el aprendizaje
  → La validación es "nueva" para el modelo en cada época""")
else:
    print("""⚠️  La curva de VALIDACIÓN está por ENCIMA de ENTRENAMIENTO
  → Esto es INUSUAL (pero puede pasar con validación pequeña)
  → Posible fluctuación aleatoria
  → O el conjunto de validación es más "fácil""")

if val_acc_trend > 0:
    print("""
✓ La accuracy de validación SIGUE SUBIENDO
  → El modelo todavía está aprendiendo
  → Podríamos entrenar más épocas
  → No hay señales fuertes de overfitting""")
elif val_acc_trend < -0.01:
    print("""
⚠️  La accuracy de validación está BAJANDO
  → Señal clara de OVERFITTING
  → El modelo empieza a memorizar en lugar de generalizar
  → Deberíamos haber parado antes (early stopping)""")
else:
    print("""
✓ La accuracy de validación se ha ESTABILIZADO
  → El modelo ha alcanzado su capacidad de aprendizaje
  → Entrenar más épocas probablemente no ayude mucho""")

print("""
🔴 CURVA DE LOSS (Pérdida):
───────────────────────────
""")

# Análisis específico de loss
if val_loss[-1] > train_loss[-1]:
    gap_description = "moderada" if loss_gap < 0.3 else "grande"
    print(f"""✓ La loss de VALIDACIÓN es mayor que la de ENTRENAMIENTO
  → Diferencia {gap_description}: {loss_gap:.3f}
  → Esto es normal (el modelo optimiza train loss)""")
    
    if loss_gap > 0.5:
        print("""  ⚠️  La diferencia es MUY GRANDE
  → Señal fuerte de OVERFITTING
  → Considerar: Dropout, regularización L2, más datos""")

if val_loss_trend < 0:
    print("""
✓ La loss de validación SIGUE BAJANDO
  → El modelo está mejorando
  → Aún hay margen para entrenar más""")
elif val_loss_trend > 0.05:
    print("""
⚠️  La loss de validación está SUBIENDO
  → OVERFITTING en progreso
  → El modelo empieza a memorizar patrones específicos
  → Momento ideal para DETENER el entrenamiento""")
else:
    print("""
✓ La loss de validación se mantiene ESTABLE
  → El modelo ha alcanzado su óptimo
  → Más entrenamiento no mejorará significativamente""")

# ========== PATRONES COMUNES ==========
print("\n" + "=" * 70)
print("📚 PATRONES COMUNES EN CURVAS DE APRENDIZAJE:")
print("=" * 70)

print("""
┌────────────────────────────────────────────────────────────────┐
│ PATRÓN 1: MODELO IDEAL                                         │
├────────────────────────────────────────────────────────────────┤
│ Train Accuracy: ↗↗↗ → Sube constantemente                     │
│ Val Accuracy:   ↗↗→ → Sube y se estabiliza cerca del train    │
│ Gap:            Pequeño (< 5%)                                 │
│                                                                │
│ Interpretación: ✅ Modelo generaliza bien                      │
│ Acción: Ninguna, ¡está perfecto!                              │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PATRÓN 2: OVERFITTING (Sobreajuste)                           │
├────────────────────────────────────────────────────────────────┤
│ Train Accuracy: ↗↗↗ → Sigue subiendo                          │
│ Val Accuracy:   ↗→↘ → Sube, se estabiliza, BAJA              │
│ Gap:            Grande y creciente (> 10%)                     │
│                                                                │
│ Interpretación: 🔴 El modelo MEMORIZA en vez de aprender      │
│ Acción:                                                        │
│   • Early stopping (parar antes)                              │
│   • Añadir Dropout                                            │
│   • Regularización L2                                         │
│   • Más datos de entrenamiento                                │
│   • Data augmentation                                         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PATRÓN 3: UNDERFITTING (Subajuste)                            │
├────────────────────────────────────────────────────────────────┤
│ Train Accuracy: ↗→ → Sube poco y se estanca BAJO             │
│ Val Accuracy:   ↗→ → Similar al train, también BAJO           │
│ Gap:            Muy pequeño, pero ambas bajas (< 60%)         │
│                                                                │
│ Interpretación: 🔴 El modelo NO tiene capacidad suficiente    │
│ Acción:                                                        │
│   • Modelo más grande (más filtros/capas)                     │
│   • Entrenar más épocas                                       │
│   • Learning rate más alto                                    │
│   • Reducir regularización                                    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PATRÓN 4: MODELO EN PROGRESO                                   │
├────────────────────────────────────────────────────────────────┤
│ Train Accuracy: ↗↗↗ → Subiendo consistentemente              │
│ Val Accuracy:   ↗↗↗ → Subiendo también                       │
│ Gap:            Moderado (5-8%)                                │
│                                                                │
│ Interpretación: 🟢 El modelo AÚN está aprendiendo             │
│ Acción:                                                        │
│   • Entrenar MÁS épocas                                       │
│   • Monitorear para detectar overfitting futuro               │
└────────────────────────────────────────────────────────────────┘
""")

# ========== DIAGNÓSTICO ESPECÍFICO DE NUESTRO MODELO ==========
print("\n" + "=" * 70)
print("🎯 DIAGNÓSTICO ESPECÍFICO DE NUESTRO MODELO:")
print("=" * 70)

# Determinar el patrón que mejor describe nuestro modelo
if final_gap < 0.05 and train_acc[-1] > 0.7:
    pattern = "✅ PATRÓN 1: MODELO IDEAL"
    recommendation = "El modelo está bien equilibrado. No se necesitan cambios importantes."
elif final_gap > 0.10 or (val_acc_trend < -0.01 and val_loss_trend > 0.05):
    pattern = "⚠️  PATRÓN 2: OVERFITTING"
    recommendation = """
    Acciones recomendadas:
    • Añadir Dropout(0.3) después de las capas Dense
    • Reducir número de épocas o usar Early Stopping
    • Implementar Data Augmentation
    • Añadir regularización L2: Dense(64, kernel_regularizer='l2')
    """
elif train_acc[-1] < 0.6 and final_gap < 0.05:
    pattern = "⚠️  PATRÓN 3: UNDERFITTING"
    recommendation = """
    Acciones recomendadas:
    • Aumentar capacidad: más filtros (Conv2D(64, 128, 256))
    • Añadir más capas convolucionales
    • Entrenar más épocas (20-30)
    • Aumentar learning rate: Adam(learning_rate=0.01)
    """
elif val_acc_trend > 0.01:
    pattern = "🟢 PATRÓN 4: MODELO EN PROGRESO"
    recommendation = """
    Acciones recomendadas:
    • Entrenar más épocas (15-20 total)
    • Monitorear para detectar cuando empiece overfitting
    • Implementar Early Stopping con patience=3
    """
else:
    pattern = "🟡 PATRÓN MIXTO"
    recommendation = "El modelo muestra características mixtas. Analizar más épocas."

print(f"\nPATRÓN IDENTIFICADO: {pattern}")
print("\nRECOMENDACIONES:")
print(recommendation)

# ========== RESUMEN EJECUTIVO ==========
print("\n" + "=" * 70)
print("📋 RESUMEN EJECUTIVO:")
print("=" * 70)

print(f"""
RESULTADOS FINALES:
───────────────────
• Train Accuracy: {train_acc[-1]*100:.2f}%
• Val Accuracy:   {val_acc[-1]*100:.2f}%
• Gap:            {final_gap*100:.2f}%

ESTADO DEL MODELO:
──────────────────
• Overfitting:  {overfitting_level}
• Underfitting: {underfitting_level}
• Convergencia: {convergence}

PATRÓN: {pattern}

GRÁFICOS GENERADOS:
───────────────────
✅ Curva de Precisión (Accuracy)
✅ Curva de Pérdida (Loss)
✅ Guardados en 'curvas_aprendizaje.png'

PRÓXIMOS PASOS:
───────────────
1. Evaluar en test set (datos nunca vistos)
2. Analizar matriz de confusión
3. Visualizar predicciones individuales
4. Si es necesario, ajustar según recomendaciones
""")

print("\n" + "=" * 70)
print("ISSUE 3(F3) COMPLETADO ✅")
print("=" * 70)
print("\n📊 Gráficos generados y análisis completo")
print("🔍 Comportamiento del modelo interpretado")
print("💡 Recomendaciones específicas proporcionadas")