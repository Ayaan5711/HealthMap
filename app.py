from flask import Flask, render_template, request
import numpy as np
from src.PredictionPipelines.predictions import ModelPipeline
from werkzeug.utils import secure_filename
from src.prediction.prediction import ImageClassification
from src.alternativedrug.AlternativeDrug import AlternateDrug
import os
from PIL import Image
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/multiple_disease_prediction')
def multiple_disease_prediction():
    return render_template('multiple-disease-prediction.html')


@app.route('/drugresponse')
def drugresponse():
    return render_template("drugresponse.html")


@app.route('/alternativedrug', methods=['GET', 'POST'])
def alternativedrug():
      if request.method == 'POST':
        selected_medicine = request.form['medicine']

        alt = AlternateDrug()
        recommendations = alt.recommendation(selected_medicine)
        # Pass the medicines data to the template
        medicines_data = alt.medicines['Drug_Name'].values.tolist()
        return render_template("alternativedrug.html", medicines=medicines_data, prediction_text=recommendations)
      else:
        # Load medicines data when the page is first loaded
        medicines_data = alt.medicines['Drug_Name'].values.tolist()
        return render_template("alternativedrug.html", medicines=medicines_data)


@app.route('/skin', methods=['GET', 'POST'])
def skin():
    return render_template('skin.html')


@app.route('/developer')
def developer():
    return render_template("developer.html")





@app.route("/malaria", methods = ['POST', 'GET'])
def malaria():
    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            model = ImageClassification(image_path=filepath)
            pred = model.malaria_predict()

        return render_template('malaria.html', prediction = pred)
    else :
        return render_template('malaria.html')


@app.route('/brain', methods=['GET','POST'])
def brain():
    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            model = ImageClassification(image_path=filepath)
            pred = model.brain_predict()

        return render_template('brain.html', prediction_text_brain = pred)
    else :
        return render_template('brain.html')
    

@app.route('/chest', methods=['GET','POST'])
def chest():
    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            model = ImageClassification(image_path=filepath)
            pred = model.chest_predict()

        return render_template('chest.html', prediction_chest = pred)
    else :
        return render_template('chest.html')




@app.route('/heart', methods=['GET', 'POST'])
def heart():

    if request.method == 'POST':

        form_data = {}
        for input_name in ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng',
                           'Oldpeak', 'Slope', 'Ca']:
            form_data[input_name] = float(request.form[input_name])
        chest_pain_type = request.form.get('ChestPain', '0')
        thal_type = request.form.get('Thal', '0')

        model = ModelPipeline("heart")
        prediction = model.heart_predict(form_data = form_data, chest_pain_type=chest_pain_type, thal_type=thal_type)
        

        return render_template('heart.html', prediction=prediction)
    return render_template('heart.html')





@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Define input features
        inputs = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Collect form data
        form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
        
        input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        model = ModelPipeline()
        prediction = model.diabetes_predict(input_data = input_array)

        return render_template('diabetes.html', prediction=prediction)
    return render_template('diabetes.html')


@app.route("/disease")
def disease():
    return render_template("disease.html")


@app.route('/breast', methods=['GET', 'POST'])
def breast():
    if request.method == 'POST':
        # Define input features
        inputs = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                  'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                  'concave_points_mean', 'symmetry_mean', 'radius_se', 
                  'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 
                  'concave_points_se', 'fractal_dimension_se', 'radius_worst', 
                  'texture_worst', 'perimeter_worst', 'area_worst', 
                  'smoothness_worst', 'compactness_worst', 'concavity_worst', 
                  'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        
        # Collect form data
        form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
        
        # Create input array for prediction
        input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        model = ModelPipeline()
        prediction = model.breast_cancer_predict(input_data = input_array)


        return render_template('breast_cancer.html', prediction=prediction)
    return render_template('breast_cancer.html')





@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        inputs = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                  'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos', 
                  'gender_Female', 'gender_Male']

        form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs[:-2]}
        gender = request.form.get('gender', 'gender_Male')

        model = ModelPipeline()
        prediction = model.liver_predict(form_data=form_data, gender=gender)


        return render_template('liver.html', prediction=prediction)
    return render_template('liver.html')




@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':

        dictionary = {
                "rbc": {"abnormal": 1, "normal": 0},
                "pc": {"abnormal": 1, "normal": 0},
                "pcc": {"present": 1, "notpresent": 0},
                "ba": {"present": 1, "notpresent": 0},
                "htn": {"yes": 1, "no": 0},
                "dm": {"yes": 1, "no": 0},
                "cad": {"yes": 1, "no": 0},
                "appet": {"good": 1, "poor": 0},
                "pe": {"yes": 1, "no": 0},
                "ane": {"yes": 1, "no": 0}
            }
        
        inputs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                  'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 
                  'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        
        form_data = {}
        for input_name in inputs:
            value = request.form.get(input_name)
            if input_name in dictionary:
                form_data[input_name] = dictionary[input_name].get(value, 0)
            else:
                form_data[input_name] = float(value)

        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        model = ModelPipeline()
        prediction = model.liver_predict(input_data)


        return render_template('kidney.html', prediction=prediction)
    return render_template('kidney.html')





@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        inputs = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                  'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                  'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
                  'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                  'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        
        # Collect form data
        form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
        
        # Create input array for prediction
        input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        model = ModelPipeline()
        prediction = model.parkinsons_predict(input_data=input_array)


        return render_template('parkinsons.html', prediction=prediction)
    return render_template('parkinsons.html')





if __name__ == '__main__':
    app.run(debug=True)
