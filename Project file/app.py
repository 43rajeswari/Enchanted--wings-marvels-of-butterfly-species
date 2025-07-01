from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# ✅ Load model (update this path to match your actual model location)
MODEL_PATH = r'C:\Users\katta\OneDrive\Desktop\prasanna\model\butterfly_model.h5'
model = load_model(MODEL_PATH)

# ✅ Define class labels (change this based on your dataset)
CLASS_NAMES = ['Monarch','swallowtail','Zebra']

# ✅ Ensure 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# ✅ Image preprocessing function
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ✅ Home page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Upload form page
@app.route('/predict')
def upload_page():
    return render_template('input.html')

# ✅ Handle prediction
@app.route('/output', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    # Save uploaded file to static folder
    path = os.path.join('static', file.filename)
    file.save(path)

    # Predict
    img = prepare_image(path)
    preds = model.predict(img)
    prediction = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Return the result and pass only the filename for image_path
    return render_template('output.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_path=file.filename)
# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
