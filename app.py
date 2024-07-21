from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from src.PredictionPipelines.predictions import ModelPipeline
from werkzeug.utils import secure_filename
from src.prediction.prediction import ImageClassification
from src.alternativedrug.AlternativeDrug import AlternateDrug
from src.Insurance.Insurance import Prediction
import os
import gdown
import warnings
from src.exception import CustomException
from src.logger import logging as lg
import time

warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

# Initialize Flask app
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
    lg.info('Rendering index page')
    return render_template("index.html")

@app.route('/home')
def home():
    lg.info('Rendering home page')
    return render_template("index.html")

@app.route("/about")
def about():
    lg.info('Rendering about page')
    return render_template("about.html")

@app.route('/contact')
def contact():
    lg.info('Rendering contact page')
    return render_template("contact.html")

@app.route('/multiple_disease_prediction')
def multiple_disease_prediction():
    lg.info('Rendering multiple disease prediction page')
    return render_template('multiple-disease-prediction.html')

@app.route('/drugresponse')
def drugresponse():
    lg.info('Rendering drug response page')
    return render_template("drugresponse.html")

@app.route('/alternativedrug', methods=['GET', 'POST'])
def alternativedrug():
    try:
        if request.method == 'POST':
            selected_medicine = request.form['medicine']
            alt = AlternateDrug()
            recommendations, medicines_data = alt.recommendation(selected_medicine)        
            lg.info(f'Recommendations for {selected_medicine} generated')
            return render_template("alternativedrug.html", medicines=medicines_data, prediction_text=recommendations)
        else:
            alt = AlternateDrug()
            medicines_data = alt.medi()
            lg.info('Rendering alternative drug page with medicines data')
            return render_template("alternativedrug.html", medicines=medicines_data)
    except CustomException as ce:
        lg.error(f'Custom error in alternativedrug route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('alternativedrug'))
    except Exception as e:
        lg.error(f'Error in alternativedrug route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('alternativedrug'))

@app.route('/skin', methods=['GET', 'POST'])
def skin():
    lg.info('Rendering skin page')
    return render_template('skin.html')

@app.route('/developer')
def developer():
    lg.info('Rendering developer page')
    return render_template("developer.html")




@app.route('/malaria', methods=['POST', 'GET'])
def malaria():
    try:
        if request.method == 'POST':
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                model = ImageClassification(image_path=filepath)
                pred = model.malaria_predict()
                lg.info('Malaria prediction made')
            return render_template('malaria.html', prediction=pred)
        else:
            return render_template('malaria.html')
    except CustomException as ce:
        lg.error(f'Custom error in malaria route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('malaria'))
    except Exception as e:
        lg.error(f'Error in malaria route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('malaria'))

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    try:
        if request.method == 'POST':
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                model = ImageClassification(image_path=filepath)
                pred = model.brain_predict()
                lg.info('Brain prediction made')
            return render_template('brain.html', prediction_text_brain=pred)
        else:
            return render_template('brain.html')
    except CustomException as ce:
        lg.error(f'Custom error in brain route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('brain'))
    except Exception as e:
        lg.error(f'Error in brain route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('brain'))

@app.route('/chest', methods=['GET', 'POST'])
def chest():
    try:
        if request.method == 'POST':
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                model = ImageClassification(image_path=filepath)
                pred = model.chest_predict()
                lg.info('Chest prediction made')
            return render_template('chest.html', prediction_chest=pred)
        else:
            return render_template('chest.html')
    except CustomException as ce:
        lg.error(f'Custom error in chest route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('chest'))
    except Exception as e:
        lg.error(f'Error in chest route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('chest'))




@app.route('/heart', methods=['GET', 'POST'])
def heart():
    try:
        if request.method == 'POST':
            form_data = {}
            for input_name in ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng',
                               'Oldpeak', 'Slope', 'Ca']:
                form_data[input_name] = float(request.form[input_name])
            chest_pain_type = request.form.get('ChestPain', '0')
            thal_type = request.form.get('Thal', '0')
            model = ModelPipeline("heart")
            prediction = model.heart_predict(form_data=form_data, chest_pain_type=chest_pain_type, thal_type=thal_type)
            lg.info('Heart prediction made')
            return render_template('heart.html', prediction=prediction)
        return render_template('heart.html')
    except CustomException as ce:
        lg.error(f'Custom error in heart route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('heart'))
    except Exception as e:
        lg.error(f'Error in heart route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('heart'))


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    try:
        if request.method == 'POST':
            inputs = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
            input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)
            model = ModelPipeline()
            prediction = model.diabetes_predict(input_data=input_array)
            lg.info('Diabetes prediction made')
            return render_template('diabetes.html', prediction=prediction)
        return render_template('diabetes.html')
    except CustomException as ce:
        lg.error(f'Custom error in diabetes route: {ce}')
        flash('A custom error occurred while processing your request.')
        return redirect(url_for('diabetes'))
    except Exception as e:
        lg.error(f'Error in diabetes route: {e}')
        flash('An error occurred while processing your request.')
        return redirect(url_for('diabetes'))



@app.route("/disease")
def disease():
    lg.info('Rendering disease page')
    return render_template("disease.html")


@app.route('/breast', methods=['GET', 'POST'])
def breast():
    try:
        if request.method == 'POST':
            inputs = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                      'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                      'concave_points_mean', 'symmetry_mean', 'radius_se', 
                      'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 
                      'concave_points_se', 'fractal_dimension_se', 'radius_worst', 
                      'texture_worst', 'perimeter_worst', 'area_worst', 
                      'smoothness_worst', 'compactness_worst', 'concavity_worst', 
                      'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
            
            form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
            input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

            model = ModelPipeline()
            prediction = model.breast_cancer_predict(input_data=input_array)
            lg.info(f'Breast cancer prediction successful: {prediction}')

            return render_template('breast_cancer.html', prediction=prediction)
    except Exception as e:
        lg.error(f'Error during breast cancer prediction: {str(e)}')
        raise CustomException(e)
    return render_template('breast_cancer.html')


@app.route('/liver', methods=['GET', 'POST'])
def liver():
    try:
        if request.method == 'POST':
            inputs = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                      'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos', 
                      'gender_Female', 'gender_Male']

            form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs[:-2]}
            gender = request.form.get('gender', 'gender_Male')

            model = ModelPipeline()
            prediction = model.liver_predict(form_data=form_data, gender=gender)
            lg.info(f'Liver disease prediction successful: {prediction}')

            return render_template('liver.html', prediction=prediction)
    except Exception as e:
        lg.error(f'Error during liver disease prediction: {str(e)}')
        raise CustomException(e)
    return render_template('liver.html')



@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    try:
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
            prediction = model.kidney_predict(input_data)
            lg.info(f'Kidney disease prediction successful: {prediction}')

            return render_template('kidney.html', prediction=prediction)
    except Exception as e:
        lg.error(f'Error during kidney disease prediction: {str(e)}')
        raise CustomException(e)
    return render_template('kidney.html')



@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    try:
        if request.method == 'POST':
            inputs = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                      'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
                      'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
                      'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                      'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
            
            form_data = {input_name: float(request.form.get(input_name, 0)) for input_name in inputs}
            input_array = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

            model = ModelPipeline()
            prediction = model.parkinsons_predict(input_data=input_array)
            lg.info(f'Parkinson\'s disease prediction successful: {prediction}')

            return render_template('parkinsons.html', prediction=prediction)
    except Exception as e:
        lg.error(f'Error during Parkinson\'s disease prediction: {str(e)}')
        raise CustomException(e)
    return render_template('parkinsons.html')




@app.route('/insurance')
def insurance():
    try:
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 
                  'Ahmedabad', 'Pune', 'Kanpur', 'Nagpur', 'Lucknow', 'Chandigarh']
        lg.info('Index page accessed successfully.')

        return render_template('insurance.html', cities=cities)
    except CustomException as ce:
        lg.error(f"CustomException in index route: {ce}")
        return "A custom error occurred while loading the index page."
    except Exception as e:
        lg.error(f"Error in index route: {e}")
        return "An error occurred while loading the index page."
    


@app.route('/insurance_predict', methods=['POST'])
def insurance_predict():
    try:
        data = request.form.to_dict()
        lg.info(f"Received data for prediction: {data}")

        ob = Prediction()
        risk_score, policy_type, policy_price = ob.predict(data)

        lg.info(f"Prediction results: Risk Score={risk_score}, Policy Type={policy_type}, Policy Price={policy_price}")
        return render_template('result.html', risk_score=risk_score, policy_type=policy_type, policy_price=policy_price)
    except CustomException as ce:
        lg.error(f"CustomException in predict route: {ce}")
        return "A custom error occurred while predicting the insurance risk."
    except Exception as e:
        lg.error(f"Error in predict route: {e}")
        return "An error occurred while predicting the insurance risk."



if __name__ == '__main__':

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
    app.run(debug=True)
