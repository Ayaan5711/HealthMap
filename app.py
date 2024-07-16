from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import json
from PIL import Image
import cv2
import warnings
import logging

# Suppress warning
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

tf.get_logger().setLevel(logging.ERROR)






app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/home')
def home():
    return render_template("index.html")



@app.route("/disease")
def disease():
    return render_template("disease.html" , symptoms_dict=symptoms_dict)


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Load the dataset and the model
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
svc = pickle.load(open('models/svc.pkl','rb'))

# Define helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']

    return desc, pre, med, die, wrkout

def predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            print("Item:", item)
            input_vector[symptoms_dict[item]] = 1
        else:
            print("Warning: symptom '{}' not found in symptoms_dict".format(item))
    return diseases_list[svc.predict([input_vector])[0]]



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptom_1 = request.form.get('symptom_1')
        symptom_2 = request.form.get('symptom_2')
        symptom_3 = request.form.get('symptom_3')
        symptom_4 = request.form.get('symptom_4')
        symptoms_list = [symptom_1, symptom_2, symptom_3, symptom_4]
        symptoms = [symptom for symptom in symptoms_list if symptom != '']

        predicted_disease = predicted_value(symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        my_precautions = []
        for i in precautions[0]:
            my_precautions.append(i)

        return render_template('disease.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout, symptoms_dict=symptoms_dict)

    return render_template('disease.html', symptoms_dict=symptoms_dict)




@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/drugresponse')
def drugresponse():
    return render_template("drugresponse.html")



 
with open('models/medicine_dict.pkl', 'rb') as f:
    medicines_dict = pickle.load(f)

medicines = pd.DataFrame(medicines_dict)

# Load similarity-vector-data from pickle in the form of dictionary
with open('models/similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

def recommendation(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = []
    for i in medicines_list:
        recommended_medicines.append({
            'medicine_name': medicines.iloc[i[0]].Drug_Name,
            'pharmeasy_link': f"https://pharmeasy.in/search/all?name={medicines.iloc[i[0]].Drug_Name}"
        })
    return recommended_medicines  


@app.route('/alternativedrug', methods=['GET', 'POST'])
def alternativedrug():
      if request.method == 'POST':
        selected_medicine = request.form['medicine']
        recommendations = recommendation(selected_medicine)
        # Pass the medicines data to the template
        medicines_data = medicines['Drug_Name'].values.tolist()
        return render_template("alternativedrug.html", medicines=medicines_data, prediction_text=recommendations)
      else:
        # Load medicines data when the page is first loaded
        medicines_data = medicines['Drug_Name'].values.tolist()
        return render_template("alternativedrug.html", medicines=medicines_data)
      


model = load_model('models/braintumor.h5')

@app.route('/brain', methods=['GET','POST'])
def brain():
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
        result = {'label': labels[indices], 'probability':round(float(probabilities) * 100)}
        return render_template("brain.html", prediction_text_brain=result)
    else:
        return render_template("brain.html")
    
def predict_disease(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


@app.route('/breast' , methods=['GET','POST'])
def breast():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict_disease(to_predict_list, to_predict_dict)
        return render_template("breast.html", prediction_text_breast=pred)
    else:
        return render_template("breast.html")

@app.route('/dibates', methods=['GET','POST'])
def dibates():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict_disease(to_predict_list, to_predict_dict)
        return render_template("dibates.html", prediction_text_dibates=pred)
    else:
        return render_template("dibates.html")

@app.route('/heart' , methods=['GET','POST'])
def heart():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict_disease(to_predict_list, to_predict_dict)
        return render_template("heart.html", prediction_text_heart=pred)
    else:
        return render_template("heart.html")

@app.route('/kidney' , methods=['GET','POST'])
def kidney():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict_disease(to_predict_list, to_predict_dict)
        return render_template("kidney.html", prediction_text_kidney=pred)
    else:
        return render_template("kidney.html")

@app.route('/liver' , methods=['GET','POST'])
def liver():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict_disease(to_predict_list, to_predict_dict)
        return render_template("liver.html", prediction_text_liver=pred)
    else:
        return render_template("liver.html")


@app.route("/malaria", methods = ['POST', 'GET'])
def malaria():
    if request.method == 'POST':
        img = Image.open(request.files['image'])
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,3))
        img = img.astype(np.float64)
        model = load_model("models/malaria.h5")
        pred = np.argmax(model.predict(img)[0])
        return render_template('malaria.html', pediction_malaria = pred)
    else :
        return render_template('malaria.html')

@app.route("/pneumonia", methods = ['POST', 'GET'])
def pneumonia():
    if request.method == 'POST':
        img = Image.open(request.files['image']).convert('L')
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,1))
        img = img / 255.0
        model = load_model("models/pneumonia.h5")
        pred = np.argmax(model.predict(img)[0])
        return render_template('pneumonia.html', pediction_pneumonia = pred)
    else :
        return render_template('pneumonia.html')




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
    return render_template('skin.html')



if __name__ == '__main__':
    app.run(debug=True)