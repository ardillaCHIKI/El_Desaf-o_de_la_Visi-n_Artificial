from flask import Flask, render_template, request, jsonify, send_file
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from PIL import Image
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Cargar modelo y datos
print("🔄 Cargando modelo...")
try:
    model = load_model('modelo_cnn_cifar10.keras')
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"⚠️  No se pudo cargar el modelo: {e}")
    model = None

# Nombres de clases
class_names = ['✈️ Avión', '🚗 Automóvil', '🐦 Pájaro', '🐱 Gato', '🦌 Ciervo',
               '🐕 Perro', '🐸 Rana', '🐴 Caballo', '🚢 Barco', '🚚 Camión']

class_names_en = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Cargar dataset CIFAR-10 para ejemplos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test_norm = x_test.astype('float32') / 255.0

def preparar_imagen(img_array):
    """Prepara una imagen para el modelo"""
    # Si es PIL Image, convertir a numpy
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # Asegurar que sea 32x32x3
    if img_array.shape != (32, 32, 3):
        img = Image.fromarray(img_array.astype('uint8'))
        img = img.resize((32, 32))
        img_array = np.array(img)
    
    # Normalizar
    img_array = img_array.astype('float32') / 255.0
    
    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def img_to_base64(img_array):
    """Convierte array numpy a base64 para mostrar en HTML"""
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype('uint8')
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Entrena el modelo primero.'}), 500
    
    try:
        # Obtener imagen del request
        if 'file' in request.files:
            # Imagen subida por el usuario
            file = request.files['file']
            img = Image.open(file.stream).convert('RGB')
            img_array = np.array(img)
        elif 'index' in request.form:
            # Imagen del dataset
            idx = int(request.form['index'])
            img_array = x_test[idx]
        else:
            return jsonify({'error': 'No se encontró imagen'}), 400
        
        # Preparar imagen
        img_prep = preparar_imagen(img_array)
        
        # Hacer predicción
        prediction = model.predict(img_prep, verbose=0)
        
        # Obtener resultados
        clase_predicha = np.argmax(prediction[0])
        confianza = float(prediction[0][clase_predicha] * 100)
        
        # Top 3 predicciones
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        top3 = [
            {
                'clase': class_names[idx],
                'clase_en': class_names_en[idx],
                'probabilidad': float(prediction[0][idx] * 100)
            }
            for idx in top3_indices
        ]
        
        # Convertir imagen a base64 para mostrar
        if img_array.shape != (32, 32, 3):
            img_display = Image.fromarray(img_array.astype('uint8')).resize((32, 32))
            img_array = np.array(img_display)
        
        img_base64 = img_to_base64(img_array)
        
        # Etiqueta real (si es del dataset)
        label_real = None
        if 'index' in request.form:
            idx = int(request.form['index'])
            label_real = class_names[y_test[idx][0]]
        
        return jsonify({
            'success': True,
            'prediccion': class_names[clase_predicha],
            'prediccion_en': class_names_en[clase_predicha],
            'confianza': confianza,
            'top3': top3,
            'imagen': img_base64,
            'label_real': label_real
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ejemplos')
def ejemplos():
    """Devuelve 20 ejemplos aleatorios del dataset"""
    indices = np.random.choice(len(x_test), 20, replace=False)
    ejemplos = []
    
    for idx in indices:
        img_array = x_test[idx]
        img_base64 = img_to_base64(img_array)
        ejemplos.append({
            'index': int(idx),
            'imagen': img_base64,
            'label': class_names[y_test[idx][0]]
        })
    
    return jsonify(ejemplos)

@app.route('/metricas')
def metricas():
    """Genera y devuelve gráfico de métricas de entrenamiento"""
    try:
        # Intentar cargar historial de entrenamiento
        import json
        with open('historial_entrenamiento.json', 'r') as f:
            history = json.load(f)
        
        # Crear gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='#4CAF50')
        ax1.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#2196F3')
        ax1.set_xlabel('Época', fontsize=12, weight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, weight='bold')
        ax1.set_title('Evolución del Accuracy', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de loss
        ax2.plot(history['loss'], label='Train Loss', linewidth=2, color='#FF5722')
        ax2.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#FF9800')
        ax2.set_xlabel('Época', fontsize=12, weight='bold')
        ax2.set_ylabel('Loss', fontsize=12, weight='bold')
        ax2.set_title('Evolución del Loss', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
        
    except Exception as e:
        # Si no hay historial, generar imagen de placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'No hay datos de entrenamiento disponibles\n\n{str(e)}',
                ha='center', va='center', fontsize=14, weight='bold')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')

@app.route('/arquitectura')
def arquitectura():
    """Muestra la arquitectura del modelo"""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        from tensorflow.keras.utils import plot_model
        
        buf = io.BytesIO()
        plot_model(model, to_file=buf, show_shapes=True, 
                  show_layer_names=True, rankdir='TB', dpi=150)
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Iniciando aplicación web...")
    print("="*70)
    print("\n📍 Accede a: http://localhost:5000")
    print("\n✨ Funcionalidades:")
    print("   • Clasificación de imágenes CIFAR-10")
    print("   • Predicción en tiempo real")
    print("   • Visualización de métricas")
    print("   • Ejemplos del dataset")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)