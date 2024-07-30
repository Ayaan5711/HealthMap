from flask import Flask, request, render_template, jsonify, url_for
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import json
import numpy as np
import os

app = Flask(__name__)

# Enable CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Set image dimensions for the model
img_height, img_width = 64, 64

# Load class labels from JSON
with open(r'server/Skin_Prediction/models/cnn_classes.json', 'r') as f:
    labelInfo = json.load(f)

# Load the pre-trained model
model = load_model(r'server/Skin_Prediction/models/skin.keras')

@app.route("/", methods=['GET'])
def index():
    # Render the upload form page
    return render_template('upload_image.html')

@app.route('/prediction', methods=['POST'])
@cross_origin()
def predictImage():
    if 'filePath' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    fileObj = request.files["filePath"]

    # Save the uploaded file to a specific directory
    upload_folder = './static/uploads'  # Ensure this is within your static folder
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    filePathName = os.path.join(upload_folder, fileObj.filename)
    fileObj.save(filePathName)

    # Open and preprocess the image
    img = Image.open(filePathName)
    img = img.resize((img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255.0
    x = x.reshape(1, img_height, img_width, 3)

    # Make a prediction
    predi = model.predict(x)
    predictedLabelIndex = np.argmax(predi[0])
    predictedLabel = labelInfo[str(predictedLabelIndex)]

    # Load additional medicine data
    with open(r'server/Skin_Prediction/models/medicine_data.json', 'r') as f:
        medicine_data = json.load(f)

    # Access specific data for the predicted label
    disease_info = medicine_data[predictedLabelIndex]
    overview = disease_info["Overview"]
    medicalTreatments = disease_info["Medical Treatments"]
    homeRemedies = disease_info["Home Remedies"]

    # Context for rendering results
    context = {
        'filePathName': f'uploads/{fileObj.filename}',  # Relative path for url_for
        'predictedLabel': predictedLabel,
        'about': overview,
        'medicine': medicalTreatments,
        'remedies': homeRemedies
    }

    # Render the results page with context
    return render_template('upload_image.html', **context)

if __name__ == "__main__":
    app.run(debug=True)
