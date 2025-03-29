import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime

# Configuration Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
IMG_SIZE = 224

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Vérification de la disponibilité du GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('Aucun GPU détecté, exécution sur CPU')
else:
    print(f'GPU détecté: {device_name}')

# Modèle CNN
def build_model():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Entraînement du modèle
def train_model():
    if os.path.exists('medical_cnn.h5'):
        return load_model('medical_cnn.h5')
    
    model = build_model()
    
    # Augmentation des données
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Chargement des données (structure: data/train/class1/, data/train/class2/, etc.)
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,  # Réduction de la taille des batchs
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,  # Réduction de la taille des batchs
        class_mode='categorical',
        subset='validation'
    )
    
    # Entraînement sans les arguments workers/use_multiprocessing pour compatibilité
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 16,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 16
    )
    
    model.save('medical_cnn.h5')
    return model

# Prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Grad-CAM
def generate_grad_cam(model, img_array, layer_name='conv2d_2'):
    # Convertir l'image en tenseur
    img_tensor = tf.convert_to_tensor(img_array)
    
    # Créer un modèle qui retourne les activations et les prédictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Calcul des gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Pooling des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplication des poids par les activations
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalisation
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Interface Flask
@app.route('/')
def home():
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prétraitement et prédiction
        img_array = preprocess_image(filepath)
        model = train_model()  # Chargement ou entraînement du modèle
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Génération Grad-CAM
        heatmap = generate_grad_cam(model, img_array)
        
        # Visualisation
        img = cv2.imread(filepath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img * 0.6
        grad_cam_path = os.path.join('static', 'grad_cam_' + filename)
        cv2.imwrite(grad_cam_path, superimposed_img)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'grad_cam': grad_cam_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return jsonify({'error': 'Invalid file type'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
