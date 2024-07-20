import pandas as pd
import pickle
import numpy as np

class Prediction:

    @staticmethod
    def predict(data):
        # Load the pre-trained model
        with open('src/models/insurance_model.pkl', 'rb') as file:
            saved_objects = pickle.load(file)
            model = saved_objects['model']
            scaler = saved_objects['scaler']
            encoder = saved_objects['encoder']
            numerical_cols = saved_objects['numerical_cols']
            categorical_cols = saved_objects['categorical_cols']  # Ensure to replace with the actual path

        data['Occupation'] = "Sedentary"

        numerical_cols = [
            'Age', 'BMI', 'Annual Income', 'Diabetes', 'Hypertension', 
            'Heart Disease', 'Cancer', 'Asthma', 'Arthritis', 'Stroke', 
            'Epilepsy', 'Kidney Disease', 'Liver Disease', 'Tuberculosis', 'HIV'
        ]
        categorical_cols = [
            'Sex', 'Smoking Status', 'Family History of Disease', 'Occupation', 'City'
        ]

        # Convert binary fields to numerical
        binary_fields = ['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Asthma', 
                            'Arthritis', 'Stroke', 'Epilepsy', 'Kidney Disease', 'Liver Disease', 
                            'Tuberculosis', 'HIV']

        for field in binary_fields:
            data[field] = 1 if data[field] == 'Yes' else 0



        df = pd.DataFrame([data])
        # Preprocess numerical columns
        X_test_numerical = scaler.transform(df[numerical_cols])

        # Preprocess categorical columns
        X_test_categorical = encoder.transform(df[categorical_cols]).toarray()

        # Combine numerical and categorical features
        X_test_processed = np.hstack((X_test_numerical, X_test_categorical))

        # Predict risk score
        risk_score = model.predict(X_test_processed)[0]
        risk_score = risk_score.round(2)

        # Determine policy type and price based on risk score
        if risk_score < 10:
            policy_type = 'Basic'
        elif risk_score < 20:
            policy_type = 'Premium'
        elif risk_score < 30:
            policy_type = 'Comprehensive'
        elif risk_score < 40:
            policy_type = 'Family Plan'
        elif risk_score < 50:
            policy_type = 'Critical Illness'
        elif risk_score < 60:
            policy_type = 'Accident Coverage'
        elif risk_score < 70:
            policy_type = 'Dental and Vision'
        else:
            policy_type = 'Long-Term Care'
        
        policy_prices = {
            'Basic': 5000,
            'Premium': 10000,
            'Comprehensive': 15000,
            'Family Plan': 20000,
            'Critical Illness': 25000,
            'Accident Coverage': 12000,
            'Dental and Vision': 8000,
            'Long-Term Care': 18000
        }

        policy_price = policy_prices[policy_type]

        return risk_score, policy_type, policy_price
