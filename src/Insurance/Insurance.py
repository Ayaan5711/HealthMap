import pickle
import numpy as np

class Insurance_Prediction:
    
    def insurance_predict(self,form_data):
        model_path = f"src/models/insurance-model.pkl"
        pca_path = f"src/pca/insurance-pca.pkl"
        scaler_path = f"src/scalar/insurance-scaler.pkl"

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)

        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        city = ['city_Ahmedabad', 'city_Bangalore','city_Chandigarh', 'city_Chennai', 'city_Delhi', 'city_Hyderabad','city_Kanpur', 'city_Kolkata', 'city_Lucknow', 'city_Mumbai','city_Nagpur', 'city_Pune']
        for i in range(len(city)):
            form_data[city[i]] = 0
            if int(form_data['city']) == i:
                form_data[i] = 1
        form_data['chronic_conditions'] = 3
        inputs = ['age', 'gender', 'occupation', 'smoking_status', 'alcohol_consumption',
       'chronic_conditions', 'previous_claims', 'income_level',
       'education_level', 'city_Ahmedabad', 'city_Bangalore',
       'city_Chandigarh', 'city_Chennai', 'city_Delhi', 'city_Hyderabad',
       'city_Kanpur', 'city_Kolkata', 'city_Lucknow', 'city_Mumbai',
       'city_Nagpur', 'city_Pune']
        
        input_data = np.array([int(form_data[input_name]) for input_name in inputs]).reshape(1, -1)
        # Apply the scaler
        scaled_data = scaler.transform(input_data)
        
        # Apply PCA
        pca_data = pca.transform(scaled_data)
        
        # Predict using the model
        risk = model.predict(pca_data)
        risk = risk[0]

        if risk < 20:
            return 'Basic Plan', 5000
        elif risk < 40:
            return 'Standard Plan', 10000
        elif risk < 60:
            return 'Premium Plan', 15000
        elif risk < 80:
            return 'Gold Plan', 20000
        else:
            return 'Platinum Plan', 25000
        # return policy, policy_price
        