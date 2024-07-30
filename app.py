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
from src.ImagePrediction.image_prediction import ImagePrediction
from src.llm_report.Report import report_generator

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
        return render_template("drugresponse.html", side_effects=side_effects)
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
    
@app.route('/multi_disease')
def multi_disease():
    return render_template("multi_disease.html")

@app.route('/disease_input_type')
def disease_input_type():
    return render_template("disease_input_type.html")




UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/disease_image_input', methods=['POST', 'GET'])
def disease_image_input():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = file_path

            model = ImagePrediction()
            pred, class_name = model.predict(img)
            llm = report_generator()
            response = llm.report(pred,class_name)
            

            return render_template("disease_image_input.html", response = response)
        return render_template("disease_image_input.html")
    return render_template("disease_image_input.html")
    

if __name__ == '__main__':
    app.run(debug=True)
    

