from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import json
import tensorflow as tf
from tensorflow import Graph
import numpy as np
import os

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

img_height, img_width = 64, 64
with open(r'server\Skin_Prediction\models\cnn_classes.json', 'r') as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)
model = load_model(r'server\Skin_Prediction\models\skin.keras')

model_graph = Graph()

@app.route("/", methods=['GET'])
def index():
    return 'Hello!'

@app.route('/prediction', methods=['POST'])
@cross_origin()
def predictImage():
    if request.method == 'POST':
        fileObj = request.files["image"]

        # Save the uploaded file to a specific directory
        upload_folder = './uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        filePathName = os.path.join(upload_folder, fileObj.filename)
        fileObj.save(filePathName)

        testimage = filePathName
        img = Image.open(testimage)  # Use Image.open() to open the image
        img = img.resize((img_height, img_width))  # Resize the image if needed
        x = image.img_to_array(img)
        x = x / 255
        x = x.reshape(1, img_height, img_width, 3)

        predi = model.predict(x)
        predictedLabel = labelInfo[(np.argmax(predi[0]))]
        dat_dict = predictedLabel
        keys_list = list(dat_dict.keys())
        keys_list = keys_list[0]

        value = dat_dict.get(keys_list, "Key not found")
        with open(r'server\Skin_Prediction\models\medicine_data.json', 'r') as f:
            medicine_data = json.load(f)

        # Assuming your JSON keys are numeric strings ("1", "2", ..., "31")

        key = list(predictedLabel.keys())[0]
        temp = medicine_data[int(key) - 1]
        print(temp)

        overview = temp["Overview"]
        medicalTreatments = temp["Medical Treatments"]
        homeRemedies = temp["Home Remedies"]

        context = {'filePathName': filePathName, 'predictedLabel': value, 'about': overview, 'medicine': medicalTreatments, 'remedies': homeRemedies}

    return context

if __name__ == "__main__":
    app.run(debug=True)
