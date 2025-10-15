
from flask import Flask, render_template_string, send_file
import io
import matplotlib.pyplot as plt
import numpy as np
from graficar_metricas_entrenamiento import graficar_metricas

app = Flask(__name__)

def entrenar_modelo():
    from entrenar_cnn_cifar10 import model, history
    return model, history

def graficar_metricas_img(history):
    from graficar_metricas_entrenamiento import graficar_metricas
    buf = io.BytesIO()
    plt.figure()
    graficar_metricas(history)
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def visualizar_clases_img():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), _ = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    np.random.seed(42)
    fig, axes = plt.subplots(10, 5, figsize=(12, 20))
    fig.suptitle('CIFAR-10: 5 Ejemplos por Clase', fontsize=16, y=0.995, weight='bold')
    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        selected_indices = np.random.choice(class_indices, 5, replace=False)
        for j, idx in enumerate(selected_indices):
            axes[i, j].imshow(x_train[idx])
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_names[i], fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

@app.route('/')
def index():
    return render_template_string('''
        <h1>Desafío de la Visión Artificial</h1>
        <ul>
            <li><a href="/metricas">Ver métricas de entrenamiento</a></li>
            <li><a href="/clases">Visualizar clases CIFAR-10</a></li>
        </ul>
    ''')

@app.route('/metricas')
def metricas():
    _, history = entrenar_modelo()
    buf = graficar_metricas_img(history)
    return send_file(buf, mimetype='image/png')

@app.route('/clases')
def clases():
    buf = visualizar_clases_img()
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
