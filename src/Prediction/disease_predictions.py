import pickle
import numpy as np

class ModelPipeline:
        
        
    def heart_predict(self, form_data):

        model_path = f"src/models/heart-model.pkl"
        pca_path = f"src/pca/heart-pca.pkl"
        scaler_path = f"src/scalar/heart-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        chest_pain_type = form_data['ChestPain']
        thal_type = form_data['Thal']

        if chest_pain_type == '0':
            form_data['ChestPain_asymptomatic'] = 1
            form_data['ChestPain_nonanginal'] = 0
            form_data['ChestPain_nontypical'] = 0
            form_data['ChestPain_typical'] = 0
        elif chest_pain_type == '1':
            form_data['ChestPain_asymptomatic'] = 0
            form_data['ChestPain_nonanginal'] = 1
            form_data['ChestPain_nontypical'] = 0
            form_data['ChestPain_typical'] = 0
        elif chest_pain_type == '2':
            form_data['ChestPain_asymptomatic'] = 0
            form_data['ChestPain_nonanginal'] = 0
            form_data['ChestPain_nontypical'] = 1
            form_data['ChestPain_typical'] = 0
        elif chest_pain_type == '3':
            form_data['ChestPain_asymptomatic'] = 0
            form_data['ChestPain_nonanginal'] = 0
            form_data['ChestPain_nontypical'] = 0
            form_data['ChestPain_typical'] = 1

        if thal_type == '0':
            form_data['Thal_fixed'] = 1
            form_data['Thal_normal'] = 0
            form_data['Thal_reversable'] = 0
        elif thal_type == '1':
            form_data['Thal_fixed'] = 0
            form_data['Thal_normal'] = 1
            form_data['Thal_reversable'] = 0
        elif thal_type == '2':
            form_data['Thal_fixed'] = 0
            form_data['Thal_normal'] = 0
            form_data['Thal_reversable'] = 1
        
        inputs = ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng',
                  'Oldpeak', 'Slope', 'Ca', 'ChestPain_asymptomatic', 'ChestPain_nonanginal',
                  'ChestPain_nontypical', 'ChestPain_typical', 'Thal_fixed', 'Thal_normal',
                  'Thal_reversable']
        
        # Create input array for prediction
        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        # Apply the scaler
        scaled_data = self.scaler.transform(input_data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict using the model
        predictions = self.model.predict(pca_data)
        prob = self.model.predict_proba(pca_data)

        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0
        

    def diabetes_predict(self, form_data):

        model_path = f"src/models/diabetes-model.pkl"
        pca_path = f"src/pca/diabetes-pca.pkl"
        scaler_path = f"src/scalar/diabetes-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        inputs = ['Pregnancies','Glucose','BloodPressure',
        	'SkinThickness',	'Insulin',	'BMI',
                	'DiabetesPedigreeFunction',	'Age']

        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)
        # Apply the scaler
        scaled_data = self.scaler.transform(input_data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict using the model
        predictions = self.model.predict(pca_data)

        prob = self.model.predict_proba(pca_data)

        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0


    def breast_cancer_predict(self, form_data):

        model_path = f"src/models/cancer-model.pkl"
        pca_path = f"src/pca/cancer-pca.pkl"
        scaler_path = f"src/scalar/cancer-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        inputs = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'radius_se',
 'perimeter_se',
 'area_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']

        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        # Apply the scaler
        scaled_data = self.scaler.transform(input_data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict using the model
        predictions = self.model.predict(pca_data)
        prob = self.model.predict_proba(pca_data)
        

        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0

        

    def liver_predict(self, form_data):

        model_path = f"src/models/liver-model.pkl"
        pca_path = f"src/pca/liver-pca.pkl"
        scaler_path = f"src/scalar/liver-scalar.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        inputs = ['age','gender','tot_bilirubin','direct_bilirubin','tot_proteins','albumin','ag_ratio','sgpt','sgot','alkphos']
        
        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        scaled_data = self.scaler.transform(input_data)

        pca_data = self.pca.transform(scaled_data)

        predictions = self.model.predict(pca_data)


        prob = self.model.predict_proba(pca_data)
        

        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0
        

    def kidney_predict(self, form_data):

        model_path = f"src/models/kidney-model.pkl"
        pca_path = f"src/pca/kidney-pca.pkl"
        scaler_path = f"src/scalar/kidney-scaler.pkl"

        inputs = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
 'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']

        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        # Apply the scaler
        scaled_data = self.scaler.transform(input_data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict using the model
        predictions = self.model.predict(pca_data)

        prob = self.model.predict_proba(pca_data)
        
        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0




    def parkinsons_predict(self, form_data):

        model_path = f"src/models/parkinsons-model.pkl"
        pca_path = f"src/pca/parkinsons-pca.pkl"
        scaler_path = f"src/scalar/parkinsons-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        inputs = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
 'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
 'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3',
 'Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
 'RPDE','DFA','spread1','spread2','D2','PPE']

        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)
        # Apply the scaler
        scaled_data = self.scaler.transform(input_data)
        
        # Apply PCA
        pca_data = self.pca.transform(scaled_data)
        
        # Predict using the model
        predictions = self.model.predict(pca_data)
        prob = self.model.predict_proba(pca_data)
        
        if prob[0][predictions[0]] > 0.70:
            return int(predictions[0])
        else:
            return 0
