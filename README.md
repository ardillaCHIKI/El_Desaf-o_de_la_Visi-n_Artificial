# El_Desaf-o_de_la_Visi-n_Artificial
Repositorio original:  
https://github.com/ardillaCHIKI/El_Desaf-o_de_la_Visi-n_Artificial.git

# El Desafío de la Visión Artificial

Este proyecto implementa una aplicación web interactiva para la clasificación de imágenes del dataset CIFAR-10 usando una red neuronal convolucional (CNN) desarrollada en TensorFlow/Keras y desplegada con Flask. Incluye scripts educativos para entender cada etapa del pipeline de Deep Learning, desde la carga y preprocesamiento de datos hasta la visualización de resultados y análisis de desempeño.

---

## Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Ejecución Paso a Paso](#ejecución-paso-a-paso)
- [Descripción de los Ficheros](#descripción-de-los-ficheros)
- [Notas y Recursos](#notas-y-recursos)

---

## Estructura del Proyecto

```
.
├── app.py
├── requirements.txt
├── README.md
├── arquitectura_modelo.json
├── modelo_cnn_cifar10.keras
├── Codigo_Demo.ipynb
├── Análisis_Teórico_CNN_vs_Visión_Tradicional/
├── scripts/
│   ├── cargar_cifar10.py
│   ├── normalizar_cifar10.py
│   ├── codificar_etiquetas_cifar10.py
│   ├── iniciar_modelo.py
│   ├── bloque_convolucional_1.py
│   ├── bloque_convolucional_2.py
│   ├── clasificador_capas_densas.py
│   ├── compilar_modelo_cnn.py
│   ├── entrenar_cnn_cifar10.py
│   ├── evaluacion_en_conjunto_de_prueba.py
│   ├── verificar_estructura_cnn.py
│   ├── visualizar_clases_cifar10.py
│   └── visualizador_de_curvas_de_aprendizaje.py
├── templates/
│   ├── index.html
│   ├── dataset.html
│   └── upload.html
├── static/
└── .vscode/
```

---

## Requisitos

- Python 3.8+
- TensorFlow, Keras, Flask, NumPy, Matplotlib, Pillow
- Instala dependencias con:
  ```sh
  pip install -r requirements.txt
  ```

---

## Ejecución Paso a Paso

1. **Instala las dependencias**  
   Ejecuta:
   ```sh
   pip install -r requirements.txt
   ```

2. **Inicia la aplicación web**  
   Desde la raíz del proyecto:
   ```sh
   python app.py
   ```
   La primera vez entrenará el modelo automáticamente (puede tardar varios minutos).

3. **Abre tu navegador**  
   Ve a [http://127.0.0.1:5000](http://127.0.0.1:5000) para acceder a la interfaz web.

4. **Explora las funcionalidades**  
   - Visualiza ejemplos del dataset.
   - Sube imágenes propias para clasificar.
   - Prueba con imágenes aleatorias del test set.
   - Consulta información del modelo entrenado.

---

## Descripción de los Ficheros

### app.py

- **Núcleo de la aplicación web Flask.**
- Carga y entrena el modelo CNN si no existe.
- Expone rutas para:
  - `/` Página principal.
  - `/dataset` Visualización de ejemplos del dataset.
  - `/upload` Subida y clasificación de imágenes.
  - `/test_sample/<int:index>` Clasificación de una imagen del test set.
  - `/random_test` Selección aleatoria de imagen de test.
  - `/model_info` Información y resumen del modelo.

### scripts/

Scripts educativos y de experimentación, cada uno aborda una etapa del pipeline:

- **[`scripts/cargar_cifar10.py`](scripts/cargar_cifar10.py):**  
  Carga el dataset CIFAR-10 y verifica dimensiones de los datos.

- **[`scripts/normalizar_cifar10.py`](scripts/normalizar_cifar10.py):**  
  Normaliza los valores de píxeles a [0, 1] y explica por qué es importante.

- **[`scripts/codificar_etiquetas_cifar10.py`](scripts/codificar_etiquetas_cifar10.py):**  
  Convierte las etiquetas a one-hot encoding y verifica la correspondencia.

- **[`scripts/iniciar_modelo.py`](scripts/iniciar_modelo.py):**  
  Inicializa un modelo `Sequential` vacío y explica su estructura.

- **[`scripts/bloque_convolucional_1.py`](scripts/bloque_convolucional_1.py):**  
  Añade el primer bloque Conv2D + MaxPooling2D, analiza dimensiones y parámetros.

- **[`scripts/bloque_convolucional_2.py`](scripts/bloque_convolucional_2.py):**  
  Añade un segundo bloque convolucional, compara y explica la progresión de features.

- **[`scripts/clasificador_capas_densas.py`](scripts/clasificador_capas_densas.py):**  
  Añade capas Flatten y Dense, completando la arquitectura para clasificación.

- **[`scripts/compilar_modelo_cnn.py`](scripts/compilar_modelo_cnn.py):**  
  Compila el modelo, explica la elección del optimizador y la función de pérdida.

- **[`scripts/entrenar_cnn_cifar10.py`](scripts/entrenar_cnn_cifar10.py):**  
  Entrena el modelo y muestra métricas de entrenamiento y validación.

- **[`scripts/evaluacion_en_conjunto_de_prueba.py`](scripts/evaluacion_en_conjunto_de_prueba.py):**  
  Evalúa el modelo en el test set, analiza resultados y sugiere mejoras.

- **[`scripts/verificar_estructura_cnn.py`](scripts/verificar_estructura_cnn.py):**  
  Verifica la estructura 2D de los datos y explica la diferencia con MLP.

- **[`scripts/visualizar_clases_cifar10.py`](scripts/visualizar_clases_cifar10.py):**  
  Visualiza ejemplos de cada clase y analiza la variabilidad del dataset.

- **[`scripts/visualizador_de_curvas_de_aprendizaje.py`](scripts/visualizador_de_curvas_de_aprendizaje.py):**  
  Genera y analiza curvas de aprendizaje (accuracy/loss), detecta overfitting/underfitting.

### templates/

- **`index.html`**: Página principal con acceso a todas las funcionalidades.
- **`dataset.html`**: Visualización de ejemplos por clase.
- **`upload.html`**: Interfaz para subir imágenes y ver resultados de clasificación.

### Otros ficheros

- **`arquitectura_modelo.json` / `modelo_cnn_cifar10.keras`**:  
  Guardan la arquitectura y pesos del modelo entrenado.
- **`Codigo_Demo.ipynb`**:  
  Notebook interactivo para pruebas rápidas y visualizaciones.
- **`requirements.txt`**:  
  Lista de dependencias necesarias.

---

## Notas y Recursos

- El código está ampliamente comentado y pensado para fines educativos.
- Puedes ejecutar cada script de la carpeta `scripts/` de forma independiente para entender cada etapa.
- El modelo y la app están preparados para experimentar y modificar fácilmente.

---
