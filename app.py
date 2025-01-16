from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Path model
MODEL_PATH = 'model/food_classification.keras'

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar kelas makanan
classes = [
    "Ayam Goreng", "Burger", "French Fries", "Gado-Gado", "Ikan Goreng",
    "Nasi Goreng", "Nasi Padang", "Mie Goreng", "Pizza", "Rawon",
    "Rendang", "Sate", "Soto"
]

# Fungsi preprocessing gambar
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# Endpoint API untuk klasifikasi makanan
@app.route('/api/klasifikasi-makanan', methods=['POST'])
def classify_food():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file:
        
        file_path = os.path.join('static/sample_images', file.filename)
        file.save(file_path)

        
        image = preprocess_image(file_path)

        
        predictions = model.predict(image)
        class_idx = np.argmax(predictions)  
        class_name = classes[class_idx]
        confidence = float(predictions[0][class_idx])

        
        os.remove(file_path)

        return jsonify({
            "predicted_class": class_name,
            "confidence": confidence
        })

    return jsonify({"error": "File processing failed"}), 500


if __name__ == '__main__':
    app.run(debug=True)
