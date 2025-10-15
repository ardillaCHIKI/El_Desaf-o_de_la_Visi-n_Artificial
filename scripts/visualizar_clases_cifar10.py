# Issue 2: Visualizar Imágenes del Dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

print("=" * 70)
print("ISSUE 2: VISUALIZACIÓN DE IMÁGENES POR CLASE")
print("=" * 70)

# Cargar el dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Nombres de las clases de CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nClases en CIFAR-10: {class_names}")
print(f"Total de imágenes de entrenamiento: {len(x_train)}")

# ========== VISUALIZACIÓN: 5 imágenes por clase ==========
print("\n📊 Generando visualización con 5 imágenes por clase...")

# Seleccionar 5 ejemplos aleatorios de cada clase
np.random.seed(42)  # Para reproducibilidad
fig, axes = plt.subplots(10, 5, figsize=(12, 20))
fig.suptitle('Dataset CIFAR-10: 5 Ejemplos por Clase (Variabilidad Intraclase)', 
              fontsize=16, y=0.995, weight='bold')

for i in range(10):  # Para cada clase
    # Encontrar todos los índices de imágenes de esta clase
    class_indices = np.where(y_train == i)[0]
    
    # Seleccionar 5 ejemplos aleatorios
    selected_indices = np.random.choice(class_indices, 5, replace=False)
    
    for j, idx in enumerate(selected_indices):
        axes[i, j].imshow(x_train[idx])
        axes[i, j].axis('off')
        
        # Añadir etiqueta solo en la primera columna
        if j == 0:
            axes[i, j].set_ylabel(class_names[i], fontsize=11, rotation=0, 
                                  labelpad=45, va='center', weight='bold')

plt.tight_layout()
plt.savefig('muestra_cifar10.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Imagen guardada como 'muestra_cifar10.png'")

# ========== ESTADÍSTICAS POR CLASE ==========
print("\n" + "=" * 70)
print("ESTADÍSTICAS DEL DATASET:")
print("=" * 70)

for class_id in range(10):
    count = np.sum(y_train == class_id)
    percentage = (count / len(y_train)) * 100
    print(f"  {class_names[class_id]:12s}: {count:5d} imágenes ({percentage:.1f}%)")

print(f"\nTotal: {len(y_train)} imágenes")

# ========== OBSERVACIONES DETALLADAS ==========
print("\n" + "=" * 70)
print("OBSERVACIONES SOBRE LA VARIABILIDAD:")
print("=" * 70)
print("""
1. FONDOS (Background):
   • Airplane: Cielos azules, nubes, algunos en pistas de aterrizaje
   • Automobile/Truck: Carreteras, estacionamientos, fondos urbanos
   • Ship: Mar abierto, puertos, con/sin olas
   • Bird/Deer: Exteriores naturales (cielo, árboles, praderas)
   • Dog/Cat/Horse: Interiores, exteriores, urbanos, rurales
   • Frog: Fondos naturales (hojas, agua, tierra)
   
2. ÁNGULOS Y PERSPECTIVAS:
   • Frontal: Automóviles, camiones, algunos animales
   • Lateral: Aviones, barcos, caballos
   • Diagonal: Pájaros en vuelo, perros corriendo
   • Desde arriba: Algunos vehículos y animales
   • Primeros planos vs. tomas lejanas
   • Objetos parcialmente visibles o completos
   
3. COLORES:
   • Airplane: Blancos, grises metálicos, algunos con colores vivos
   • Automobile/Truck: Rojos, azules, blancos, negros, multicolor
   • Bird: Variedad amplia (azules, rojos, pardos, negros)
   • Cat/Dog: Marrones, negros, blancos, grises, atigrados
   • Deer/Horse: Marrones predominantes, con variaciones
   • Frog: Verdes, marrones, algunos con colores vivos
   • Ship: Blancos, grises, colores de carga
   
4. FORMAS Y POSES:
   • Animales: De pie, sentados, acostados, en movimiento, volando
   • Vehículos: Diferentes orientaciones, modelos, tamaños
   • Bird: Posados vs. en vuelo
   • Dog/Cat: Diversas razas con diferentes proporciones corporales
   • Horse/Deer: Diferentes ángulos de cabeza y cuerpo
   
5. CONDICIONES DE ILUMINACIÓN:
   • Día soleado vs. nublado
   • Interiores con luz artificial
   • Sombras variables
   • Contraste alto vs. bajo
   
6. CALIDAD Y RESOLUCIÓN:
   • Algunas imágenes más nítidas que otras
   • Variabilidad en enfoque y claridad
   • Compresión visible en algunas imágenes (32x32 es pequeño)

DESAFÍOS PARA LA CLASIFICACIÓN:
--------------------------------
- Alta variabilidad INTRACLASE (dentro de la misma clase)
- Similitud INTERCLASE en algunos casos:
  - Cat vs. Dog: Ambos cuadrúpedos, similar tamaño
  - Automobile vs. Truck: Ambos vehículos terrestres
  - Deer vs. Horse: Similar forma corporal
- Imágenes de baja resolución (32x32 píxeles)
- Oclusiones parciales de objetos
- Fondos complejos que pueden confundir al modelo

Esta diversidad hace que CIFAR-10 sea un benchmark realista y 
desafiante para evaluar el rendimiento de redes convolucionales.
""")

print("\n" + "=" * 70)
print("✅ ISSUE 2 COMPLETADO")
print("=" * 70)