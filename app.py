from flask import Flask, request, render_template
import numpy as np
import cv2
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import warnings
import json
import logging

# Suppress warning
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

tf.get_logger().setLevel(logging.ERROR)

from src.exception import CustomException
from src.logger import logging as lg
from src.disease_prediction.disease_prediction import DiseasePrediction
from src.alternativedrug.AlternativeDrug import AlternateDrug
from src.Prediction.disease_predictions import ModelPipeline
from src.Insurance.Insurance import Insurance_Prediction

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("index.html")

with open('src/datasets/symptoms.json', 'r') as json_file:
    symptoms_dict = json.load(json_file)
with open('src/datasets/disease_list.json', 'r') as json_file:
    diseases_list = json.load(json_file)

@app.route("/disease")
def disease():
    return render_template("disease.html", symptoms_dict=symptoms_dict)

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            symptom_1 = request.form.get('symptom_1')
            symptom_2 = request.form.get('symptom_2')
            symptom_3 = request.form.get('symptom_3')
            symptom_4 = request.form.get('symptom_4')
            symptoms_list = [symptom_1, symptom_2, symptom_3, symptom_4]
            
            model = DiseasePrediction()
            predicted_disease, dis_des, my_precautions, medications, rec_diet, workout, symptoms_dict = model.predict(symptoms_list=symptoms_list)

            return render_template('disease.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout, symptoms_dict=symptoms_dict)
        return render_template('disease.html', symptoms_dict=symptoms_dict)
    except Exception as e:
        lg.error(f"Error in /predict route: {e}")
        raise CustomException(e, sys)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/drugresponse', methods=['GET', 'POST'])
def drugresponse():
    try:
        with open('src/datasets/side_effects.json', 'r') as file:
            side_effects_data = json.load(file)
        side_effects = None
        if request.method == 'POST':
            drug_name = request.form.get('drug_name')
            side_effects = side_effects_data.get(drug_name, "No data available for this drug.")
        return render_template("drugresponse.html", side_effects=side_effects[:10])
    except Exception as e:
        lg.error(f"Error in /drugresponse route: {e}")
        raise CustomException(e, sys)

@app.route('/alternativedrug', methods=['GET', 'POST'])
def alternativedrug():
    try:
        if request.method == 'POST':
            selected_medicine = request.form['medicine']
            alt = AlternateDrug()
            recommendations, medicines_data = alt.recommendation(selected_medicine)  
            return render_template("alternativedrug.html", medicines=medicines_data, prediction_text=recommendations)
        else:
            alt = AlternateDrug()
            medicines_data = alt.medi()
            return render_template("alternativedrug.html", medicines=medicines_data)
    except Exception as e:
        lg.error(f"Error in /alternativedrug route: {e}")
        raise CustomException(e, sys)

@app.route('/liver', methods=['GET', 'POST'])
def liver():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.liver_predict(to_predict_dict)
            return render_template("liver.html", prediction_text_liver=pred)
        else:
            return render_template("liver.html")
    except Exception as e:
        lg.error(f"Error in /liver route: {e}")
        raise CustomException(e, sys)

@app.route('/breast', methods=['GET', 'POST'])
def breast():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.breast_cancer_predict(to_predict_dict)
            return render_template("breast.html", prediction_text=pred)
        else:
            return render_template("breast.html")
    except Exception as e:
        lg.error(f"Error in /breast route: {e}")
        raise CustomException(e, sys)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.diabetes_predict(to_predict_dict)
            return render_template("diabetes.html", prediction_text=pred)
        else:
            return render_template("diabetes.html")
    except Exception as e:
        lg.error(f"Error in /diabetes route: {e}")
        raise CustomException(e, sys)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.heart_predict(form_data=to_predict_dict)
            return render_template("heart.html", prediction_text=pred)
        else:
            return render_template("heart.html")
    except Exception as e:
        lg.error(f"Error in /heart route: {e}")
        raise CustomException(e, sys)

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.kidney_predict(to_predict_dict)
            return render_template("kidney.html", prediction_text=pred)
        else:
            return render_template("kidney.html")
    except Exception as e:
        lg.error(f"Error in /kidney route: {e}")
        raise CustomException(e, sys)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            model = ModelPipeline()
            pred = model.parkinsons_predict(to_predict_dict)
            return render_template("parkinsons.html", prediction_text=pred)
        else:
            return render_template("parkinsons.html")
    except Exception as e:
        lg.error(f"Error in /parkinsons route: {e}")
        raise CustomException(e, sys)

@app.route('/insurance', methods=['GET', 'POST'])
def insurance():
    try:
        if request.method == 'POST':
            form_data = request.form.to_dict()
            model = Insurance_Prediction()
            policy, policy_price = model.insurance_predict(form_data=form_data)
            return render_template("insurance.html", policy=policy, policy_price=policy_price)
        else:
            return render_template("insurance.html")
    except Exception as e:
        lg.error(f"Error in /insurance route: {e}")
        raise CustomException(e, sys)

@app.route("/malaria", methods=['POST', 'GET'])
def malaria():
    try:
        if request.method == 'POST':
            img = Image.open(request.files['image'])
            img = img.resize((36,36))
            img = np.asarray(img)
            img = img.reshape((1,36,36,3))
            img = img.astype(np.float64)
            model = load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('malaria.html', pediction_malaria=pred)
        else:
            return render_template('malaria.html')
    except Exception as e:
        lg.error(f"Error in /malaria route: {e}")
        raise CustomException(e, sys)

@app.route("/pneumonia", methods=['POST', 'GET'])
def pneumonia():
    try:
        if request.method == 'POST':
            img = Image.open(request.files['image']).convert('L')
            img = img.resize((36,36))
            img = np.asarray(img)
            img = img.reshape((1,36,36,1))
            img = img / 255.0
            model = load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('pneumonia.html', pediction_pneumonia=pred)
        else:
            return render_template('pneumonia.html')
    except Exception as e:
        lg.error(f"Error in /pneumonia route: {e}")
        raise CustomException(e, sys)

model = load_model('models/braintumor.h5')

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    try:
        if request.method == 'POST':
            img = request.files['image']
            img_bytes = img.read()
            img_array = np.array(bytearray(img_bytes), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (300, 300))
            img_array = np.expand_dims(img, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)
            predictions = model.predict(img_array)
            indices = np.argmax(predictions)
            probabilities = np.max(predictions)
            labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
            result = {'label': labels[indices], 'probability': round(float(probabilities) * 100)}
            return render_template("brain.html", prediction_text_brain=result)
        else:
            return render_template("brain.html")
    except Exception as e:
        lg.error(f"Error in /brain route: {e}")
        raise CustomException(e, sys)

def get_treatment(path):
    with open(path) as f:
        return json.load(f)

treatment_dict = get_treatment("skin_disorder.json")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def is_skin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = np.sum(mask > 0)
    skin_percent = skin_pixels / (img.shape[0] * img.shape[1]) * 100
    return skin_percent > 5

@app.route('/skin', methods=['GET', 'POST'])
def skin():
    try:
        return render_template('skin.html')
    except Exception as e:
        lg.error(f"Error in /skin route: {e}")
        raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        lg.error(f"Error starting Flask app: {e}")
        raise CustomException(e, sys)

