import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize app
app = Flask(__name__)

# Load the trained model
model = load_model('model/Tiger_classifier.h5')

# Define class labels in the same order used during training
class_labels = ['cheetah', 'leopard', 'lion', 'puma', 'tiger']

# Define a confidence threshold
threshold = 0.48

# Prediction function
def predict_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]

        if confidence < threshold:
            return "Invalid image (Please provide image from tiger family)", confidence
        else:
            return class_labels[predicted_index], confidence

    except Exception as e:
        return "Invalid image (Error in processing)", 0.0

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', result="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', result="No selected file")

        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        prediction, confidence = predict_image(filepath)
        result = f"{prediction} ({confidence*100:.2f}%)"
        return render_template('index.html', result=result, image_path=filepath)

    return render_template('index.html', result=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
