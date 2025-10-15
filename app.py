# app.py - Aplicación Flask para Clasificación de Imágenes CIFAR-10
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Nombres de las clases CIFAR-10
CLASS_NAMES = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
               'perro', 'rana', 'caballo', 'barco', 'camión']

CLASS_NAMES_EN = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Variables globales
model = None
x_train = None
y_train = None
x_test = None
y_test = None

# ========== FUNCIONES AUXILIARES ==========

def load_and_train_model():
    """Cargar datos y entrenar el modelo"""
    global model, x_train, y_train, x_test, y_test
    
    print("Cargando dataset CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalizar
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    # One-hot encoding
    y_train_oh = to_categorical(y_train, 10)
    y_test_oh = to_categorical(y_test, 10)
    
    print("Construyendo modelo...")
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
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Entrenando modelo (10 épocas)...")
    model.fit(x_train_norm, y_train_oh, 
              epochs=10, 
              batch_size=64, 
              validation_split=0.1,
              verbose=1)
    
    print("Evaluando en test set...")
    test_loss, test_acc = model.evaluate(x_test_norm, y_test_oh, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    return test_acc

def image_to_base64(img_array):
    """Convertir array numpy a base64 para mostrar en HTML"""
    # Asegurar que esté en rango 0-255
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def preprocess_uploaded_image(file):
    """Preprocesar imagen subida por el usuario"""
    # Leer imagen
    img = Image.open(file.stream)
    
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar a 32x32
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convertir a array numpy
    img_array = np.array(img)
    
    # Normalizar
    img_normalized = img_array.astype('float32') / 255.0
    
    return img_array, img_normalized

def predict_image(img_normalized):
    """Predecir clase de una imagen"""
    # Añadir dimensión de batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predecir
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Obtener top 3 predicciones
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3_probs = predictions[top3_indices]
    
    results = []
    for idx, prob in zip(top3_indices, top3_probs):
        results.append({
            'class_id': int(idx),
            'class_name': CLASS_NAMES[idx],
            'class_name_en': CLASS_NAMES_EN[idx],
            'probability': float(prob),
            'percentage': f"{prob*100:.2f}"
        })
    
    return results

# ========== RUTAS DE LA APLICACIÓN ==========

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/dataset')
def dataset():
    """Ver ejemplos del dataset"""
    # Seleccionar 5 imágenes aleatorias de cada clase
    examples = []
    np.random.seed(42)
    
    for class_id in range(10):
        # Encontrar índices de esta clase
        class_indices = np.where(y_train == class_id)[0]
        
        # Seleccionar 5 aleatorias
        selected = np.random.choice(class_indices, 5, replace=False)
        
        # Convertir a base64
        images_b64 = [image_to_base64(x_train[idx]) for idx in selected]
        
        examples.append({
            'class_id': int(class_id),
            'class_name': CLASS_NAMES[class_id],
            'class_name_en': CLASS_NAMES_EN[class_id],
            'images': images_b64
        })
    
    return render_template('dataset.html', examples=examples)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Subir y clasificar imagen"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    # POST: procesar imagen
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    try:
        # Procesar imagen
        img_original, img_normalized = preprocess_uploaded_image(file)
        
        # Predecir
        predictions = predict_image(img_normalized)
        
        # Convertir imagen a base64 para mostrar
        img_b64_original = image_to_base64(img_original)
        
        # Retornar resultados
        return jsonify({
            'success': True,
            'image': img_b64_original,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_sample/<int:index>')
def test_sample(index):
    """Clasificar una imagen del test set"""
    if index < 0 or index >= len(x_test):
        return jsonify({'error': 'Índice fuera de rango'}), 400
    
    # Obtener imagen
    img = x_test[index]
    img_normalized = img.astype('float32') / 255.0
    
    # Predecir
    predictions = predict_image(img_normalized)
    
    # Clase real
    real_class_id = int(y_test[index][0])
    
    # Convertir a base64
    img_b64 = image_to_base64(img)
    
    return jsonify({
        'image': img_b64,
        'predictions': predictions,
        'real_class_id': real_class_id,
        'real_class_name': CLASS_NAMES[real_class_id]
    })

@app.route('/random_test')
def random_test():
    """Obtener índice aleatorio del test set"""
    random_index = np.random.randint(0, len(x_test))
    return jsonify({'index': int(random_index)})

@app.route('/model_info')
def model_info():
    """Información del modelo"""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    # Obtener resumen del modelo
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    model_summary = string_buffer.getvalue()
    
    return jsonify({
        'total_params': int(model.count_params()),
        'summary': model_summary,
        'architecture': [
            {'name': layer.name, 'type': layer.__class__.__name__}
            for layer in model.layers
        ]
    })

# ========== INICIALIZACIÓN ==========

@app.before_request
def initialize():
    """Inicializar modelo antes de la primera petición"""
    global model
    if model is None:
        print("\n" + "="*70)
        print("INICIALIZANDO APLICACIÓN")
        print("="*70)
        load_and_train_model()
        print("="*70)
        print("APLICACIÓN LISTA")
        print("="*70 + "\n")

# ========== TEMPLATES HTML ==========

def create_templates():
    """Crear carpeta templates y archivos HTML"""
    os.makedirs('templates', exist_ok=True)
    
    # ========== index.html ==========
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN CIFAR-10 - Clasificador de Imágenes</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border-color: #667eea;
        }
        
        .card-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .card p {
            color: #666;
            line-height: 1.6;
        }
        
        .btn {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .classes {
            margin-top: 40px;
            padding: 30px;
            background: #f5f5f5;
            border-radius: 15px;
        }
        
        .classes h2 {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        
        .class-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
        }
        
        .class-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .class-item:hover {
            border-color: #667eea;
            transform: scale(1.05);
        }
        
        @media (max-width: 768px) {
            .class-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 CNN CIFAR-10</h1>
            <p class="subtitle">Clasificador de Imágenes con Deep Learning</p>
        </header>
        
        <div class="content">
            <div class="cards">
                <div class="card" onclick="location.href='/dataset'">
                    <div class="card-icon">📊</div>
                    <h2>Ver Dataset</h2>
                    <p>Explora ejemplos de las 50,000 imágenes de entrenamiento de CIFAR-10</p>
                    <a href="/dataset" class="btn">Ver Ejemplos</a>
                </div>
                
                <div class="card" onclick="location.href='/upload'">
                    <div class="card-icon">📤</div>
                    <h2>Subir Imagen</h2>
                    <p>Sube tu propia imagen y el modelo CNN la clasificará en una de las 10 categorías</p>
                    <a href="/upload" class="btn">Clasificar Imagen</a>
                </div>
            </div>
            
            <div class="classes">
                <h2>📋 Clases de CIFAR-10</h2>
                <div class="class-grid">
                    {% for class_name in class_names %}
                    <div class="class-item">
                        <strong>{{ loop.index0 }}</strong>: {{ class_name }}
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>''')
    
    # ========== dataset.html ==========
    with open('templates/dataset.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset CIFAR-10</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .back-btn {
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .content {
            padding: 40px;
        }
        
        .class-section {
            margin-bottom: 50px;
            padding: 30px;
            background: #f9f9f9;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .class-header {
            margin-bottom: 20px;
        }
        
        .class-title {
            font-size: 1.8em;
            color: #333;
            margin-bottom: 5px;
        }
        
        .class-subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .images-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
        }
        
        .image-box {
            background: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .image-box:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .image-box img {
            width: 100%;
            height: auto;
            border-radius: 5px;
            image-rendering: pixelated;
        }
        
        @media (max-width: 768px) {
            .images-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <a href="/" class="back-btn">← Volver al Inicio</a>
            <h1>📊 Dataset CIFAR-10</h1>
            <p>5 ejemplos aleatorios de cada clase (50,000 imágenes totales)</p>
        </header>
        
        <div class="content">
            {% for example in examples %}
            <div class="class-section">
                <div class="class-header">
                    <h2 class="class-title">{{ example.class_id }}: {{ example.class_name|capitalize }}</h2>
                    <p class="class-subtitle">{{ example.class_name_en }}</p>
                </div>
                <div class="images-grid">
                    {% for image in example.images %}
                    <div class="image-box">
                        <img src="{{ image }}" alt="{{ example.class_name }}">
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>''')
    
    # ========== upload.html ==========
    with open('templates/upload.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificar Imagen</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .back-btn {
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            text-align: center;
            padding: 40px;
            background: #f9f9f9;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }
        
        .upload-area:hover {
            background: #f0f0f0;
            border-color: #764ba2;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .results {
            display: none;
            padding: 30px;
            background: #f9f9f9;
            border-radius: 15px;
        }
        
        .results.show {
            display: block;
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            align-items: start;
        }
        
        .image-preview {
            text-align: center;
        }
        
        .image-preview img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            image-rendering: pixelated;
        }
        
        .predictions {
            background: white;
            padding: 20px;
            border-radius: 10px;
        }
        
        .prediction-item {
            margin-bottom: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .prediction-item.top {
            background: #e8f5e9;
            border-left-color: #4caf50;
        }
        
        .prediction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .class-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .probability {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .probability-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .probability-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
        }
        
        .test-section {
            margin-top: 30px;
            padding: 20px;
            background: #fff3cd;
            border-radius: 10px;
            text-align: center;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .result-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <a href="/" class="back-btn">← Volver al Inicio</a>
            <h1>📤 Clasificar Imagen</h1>
            <p>Sube tu imagen o prueba con ejemplos del dataset</p>
        </header>
        
        <div class="content">
            <div class="upload-section">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <h2>Click para seleccionar imagen</h2>
                    <p>o arrastra y suelta aquí</p>
                    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">Formatos: JPG, PNG (se redimensionará a 32x32)</p>
                </div>
                <input type="file" id="fileInput" accept="image/*">
                <button class="btn" id="uploadBtn" disabled>Clasificar Imagen</button>
            </div>
            
            <div class="test-section">
                <h3>🎲 ¿No tienes imagen? Prueba con el dataset</h3>
                <button class="btn" id="randomTestBtn">Clasificar Imagen Aleatoria del Test Set</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">Clasificando...</p>
            </div>
            
            <div class="results" id="results">
                <h2 style="margin-bottom: 20px;">🎯 Resultados de la Clasificación</h2>
                <div class="result-grid">
                    <div class="image-preview">
                        <h3>Imagen Original</h3>
                        <img id="resultImage" src="" alt="Imagen">
                    </div>
                    <div class="predictions" id="predictions">
                        <!-- Se llenará dinámicamente -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // Manejo de selección de archivo
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.getElementById('uploadBtn').disabled = false;
            }
        });
        
        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.background = '#f0f0f0';
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            this.style.background = 'white';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.background = 'white';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                document.getElementById('fileInput').files = files;
                document.getElementById('uploadBtn').disabled = false;
            }
        });
        
        // Subir y clasificar
        document.getElementById('uploadBtn').addEventListener('click', function() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            // Mostrar loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Mostrar resultados
                displayResults(data.image, data.predictions, null);
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                alert('Error al clasificar imagen: ' + error);
            });
        });
        
        // Probar con imagen aleatoria del test set
        document.getElementById('randomTestBtn').addEventListener('click', function() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            
            // Obtener índice aleatorio
            fetch('/random_test')
            .then(response => response.json())
            .then(data => {
                // Clasificar esa imagen
                return fetch('/test_sample/' + data.index);
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Mostrar resultados con clase real
                displayResults(data.image, data.predictions, {
                    id: data.real_class_id,
                    name: data.real_class_name
                });
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                alert('Error: ' + error);
            });
        });
        
        // Función para mostrar resultados
        function displayResults(imageBase64, predictions, realClass) {
            // Mostrar imagen
            document.getElementById('resultImage').src = imageBase64;
            
            // Crear HTML de predicciones
            let html = '';
            
            predictions.forEach((pred, index) => {
                const isTop = index === 0;
                const isCorrect = realClass && pred.class_id === realClass.id;
                
                let itemClass = 'prediction-item';
                if (isTop) itemClass += ' top';
                
                html += `
                    <div class="${itemClass}">
                        <div class="prediction-header">
                            <span class="class-name">
                                ${index + 1}. ${pred.class_name}
                                ${isTop ? '🏆' : ''}
                                ${isCorrect ? '✅' : ''}
                            </span>
                            <span class="probability">${pred.percentage}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${pred.percentage}%"></div>
                        </div>
                        <p style="margin-top: 5px; color: #666; font-size: 0.9em;">
                            ${pred.class_name_en}
                        </p>
                    </div>
                `;
            });
            
            // Añadir información de clase real si existe
            if (realClass) {
                const isCorrect = predictions[0].class_id === realClass.id;
                html = `
                    <div style="padding: 15px; background: ${isCorrect ? '#e8f5e9' : '#ffebee'}; 
                                border-radius: 10px; margin-bottom: 20px;">
                        <strong style="font-size: 1.2em;">
                            Clase Real: ${realClass.name}
                            ${isCorrect ? '✅ ¡Correcto!' : '❌ Incorrecto'}
                        </strong>
                    </div>
                ` + html;
            }
            
            document.getElementById('predictions').innerHTML = html;
            document.getElementById('results').classList.add('show');
            
            // Scroll suave a resultados
            document.getElementById('results').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest' 
            });
        }
    </script>
</body>
</html>''')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PREPARANDO APLICACIÓN FLASK")
    print("="*70)
    print("\nCreando templates HTML...")
    create_templates()
    print("✅ Templates creados en carpeta 'templates/'")
    
    print("\n" + "="*70)
    print("INSTRUCCIONES:")
    print("="*70)
    print("""
1. El modelo se entrenará automáticamente al iniciar la app (10 épocas)
2. Esto tomará varios minutos la primera vez
3. Una vez listo, abre tu navegador en: http://127.0.0.1:5000
4. Podrás:
   - Ver ejemplos del dataset CIFAR-10
   - Subir tus propias imágenes para clasificar
   - Probar con imágenes aleatorias del test set
   
NOTA: La app redimensionará automáticamente cualquier imagen a 32x32
      (tamaño requerido por CIFAR-10)
    """)
    
    print("\n" + "="*70)
    print("INICIANDO SERVIDOR FLASK...")
    print("="*70 + "\n")
    
    # Iniciar aplicación
    app.run(debug=True, host='0.0.0.0', port=5000)