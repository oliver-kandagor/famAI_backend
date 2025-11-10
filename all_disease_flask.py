from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

# *******************
# للتأكد من أن Keras / TensorFlow يعملان بالإصدار الصحيح لديك
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ----------------------------
# 1. النموذج الأول (apple_disease_model.h5)
# ----------------------------
MODEL1_PATH = r"D:\CS.4\Graduation Project\Project\flask_server\apple_disease_model.h5"
model1 = tf.keras.models.load_model(MODEL1_PATH)

class_labels_model1 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Not_plant"
]

# ----------------------------
# 2. إعادة بناء النموذج الثاني (38 فئة) وتحميل الأوزان
# ----------------------------
# بناء بنية الـ CNN كما في كودك بالضبط:

classifier = Sequential()

# Convolution Step 1
classifier.add(Conv2D(
    filters=96,
    kernel_size=(11, 11),
    strides=(4, 4),
    padding='valid',
    activation='relu',
    input_shape=(224, 224, 3)
))
# Max Pooling Step 1 + BatchNorm
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Conv2D(
    filters=256,
    kernel_size=(11, 11),
    strides=(1, 1),
    padding='valid',
    activation='relu'
))
# Max Pooling Step 2 + BatchNorm
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3 + BatchNorm
classifier.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    activation='relu'
))
classifier.add(BatchNormalization())

# Convolution Step 4 + BatchNorm
classifier.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    activation='relu'
))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    activation='relu'
))
# Max Pooling Step 3 + BatchNorm
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Flatten + Full Connection + Dropout + BatchNorm
classifier.add(Flatten())

classifier.add(Dense(units=4096, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())

classifier.add(Dense(units=4096, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())

classifier.add(Dense(units=1000, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

# طبقة الإخراج: 38 فئة + Softmax
classifier.add(Dense(units=38, activation='softmax'))

# الآن نحمّل الأوزان التي لديك في best_weights_9.hdf5
MODEL2_WEIGHTS_PATH = r"D:\CS.4\Graduation Project\Project\flask_server\best_weights_9.hdf5"
classifier.load_weights(MODEL2_WEIGHTS_PATH)

# نُسميه model2 للاستخدام لاحقًا
model2 = classifier

# قائمة الفئات للنموذج الثاني (تأكد من ترتيبها مطابق للترتيب الذي تدربت عليه)
class_labels_model2 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)__healthy",
    "Cherry(including_sour)__Powdery_mildew",
    "Corn(maize)__Cercospora_leaf_spot",
    "Corn(maize)_Common_rust",
    "Corn(maize)__healthy",
    "Corn(maize)Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# ----------------------------
# 3. دالة تحضير الصورة
# ----------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    تحضير الصورة بنفس خطوات ما قبل المعالجة:
    - تحويل إلى RGB
    - تغيير حجمها إلى 224×224
    - تحويلها إلى مصفوفة numpy وتوسيع بعد batch dimension
    - تطبيع القيم إلى [0, 1]
    """
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ----------------------------
# 4. نقطة النهاية /predict
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        processed_image = preprocess_image(img)

        # ======== الخطوة الأولى: النموذج الأول (Not_plant أو أمراض التفاح) ========
        preds1 = model1.predict(processed_image)  # شكل (1, 5)
        idx1 = np.argmax(preds1[0])
        class1 = class_labels_model1[idx1]
        confidence1 = float(preds1[0][idx1])

        # حدّ الثقة لاكتشاف "Not_plant"
        confidence_threshold = 0.5
        if class1 == "Not_plant" and confidence1 >= confidence_threshold:
            return jsonify({
                'disease': "Not_plant",
                'confidence': confidence1
            })

        # ======== الخطوة الثانية: النموذج الثاني (38 فئة نباتية) ========
        preds2 = model2.predict(processed_image)  # شكل (1, 38)
        idx2 = np.argmax(preds2[0])
        class2 = class_labels_model2[idx2]
        confidence2 = float(preds2[0][idx2])

        return jsonify({
            'disease': class2,
            'confidence': confidence2
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
