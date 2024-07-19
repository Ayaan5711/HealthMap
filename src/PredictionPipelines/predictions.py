import pickle
import numpy as np

class ModelPipeline:
        

    def heart_predict(self, form_data,chest_pain_type, thal_type):

        model_path = f"src/models/heart-model.pkl"
        pca_path = f"src/pca/heart-pca.pkl"
        scaler_path = f"src/scalar/heart-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

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

        if predictions == 0:
            return  "You might not have Atherosclerotic Heart Disease."
        else:
            return "You might have Atherosclerotic Heart Disease."
        

    def diabetes_predict(self, input_data):

        model_path = f"src/models/diabetes-model.pkl"
        pca_path = f"src/pca/diabetes-pca.pkl"
        scaler_path = f"src/scalar/diabetes-scaler.pkl"

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

        if predictions == 0:
            return  "You might not have Diabetes."
        else:
            return "You might have Diabetes."


    def breast_cancer_predict(self, input_data):

        model_path = f"src/models/cancer-model.pkl"
        pca_path = f"src/pca/cancer-pca.pkl"
        scaler_path = f"src/scalar/cancer-scaler.pkl"

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


        if predictions == 0:
            return  "You might have Benign Breast Cancer."
        else:
            return "You might have Maligant Breast Cancer."
        

    def liver_predict(self, form_data, gender):

        model_path = f"src/models/liver-model.pkl"
        pca_path = f"src/pca/liver-pca.pkl"
        scaler_path = f"src/scalar/liver-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            self.pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        inputs = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 
                  'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos', 
                  'gender_Female', 'gender_Male']

        gender_mapping = {'gender_Female': [1, 0], 'gender_Male': [0, 1]}
        gender_encoded = gender_mapping[gender]
        form_data.update(dict(zip(['gender_Female', 'gender_Male'], gender_encoded)))
        
        input_data = np.array([form_data[input_name] for input_name in inputs]).reshape(1, -1)

        scaled_data = self.scaler.transform(input_data)

        pca_data = self.pca.transform(scaled_data)

        predictions = self.model.predict(pca_data)


        if predictions == 0:
            return  "You may not have Liver Disease."
        else:
            return "You may have Liver Disease."
        

    def kidney_predict(self, input_data):

        model_path = f"src/models/kidney-model.pkl"
        pca_path = f"src/pca/kidney-pca.pkl"
        scaler_path = f"src/scalar/kidney-scaler.pkl"

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


        if predictions == 0:
            return  "You might not have Kidney Disease."
        else:
            return "You might have Kidney Disease."


    def parkinsons_predict(self, input_data):

        model_path = f"src/models/parkinsons-model.pkl"
        pca_path = f"src/pca/parkinsons-pca.pkl"
        scaler_path = f"src/scalar/parkinsons-scaler.pkl"

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


        if predictions == 0:
            return  "You might not have Parkinsons Disease."
        else:
            return "You might have Parkinsons Disease."
