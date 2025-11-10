from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


MODEL_PATH = r"D:\CS.4\Graduation Project\AI Model\Model with 90 accuracy\apple_disease_model.h5"
model = load_model(MODEL_PATH)

class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Not_plant"
]

def preprocess_image(img):
    """تحضير الصورة قبل إدخالها للنموذج"""
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # تطبيع الصورة بين 0 و 1
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        processed_image = preprocess_image(img)
        predictions = model.predict(processed_image)

        # الحصول على الفئة الأعلى احتمالية
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        # تحديد حد أدنى للثقة، إذا كان أقل من 50% نقول "غير معروف"
        confidence_threshold = 0.5
        if confidence < confidence_threshold:
            return jsonify({'disease': 'Unknown', 'confidence': confidence})

        predicted_class = class_labels[predicted_class_index]

        return jsonify({
            'disease': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
