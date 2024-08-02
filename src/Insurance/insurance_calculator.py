import google.generativeai as genai
import os
from dotenv import load_dotenv


def insurance(data):

    data = data.split(",")

    prompt = f"""I need to calculate insurance prices based on various factors associated with different diseases. 
For each disease listed ,{data} please provide a factor that represents how much the presence of that disease should increase the insurance price.
Consider a scale where higher numbers indicate a higher impact on the insurance price.
just give list of numbers nothing else, without index

"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key) 
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([prompt])
    response = response.text
    response = response.strip("\n")
    response = response.split(",") 
    ans = []
    for i in response:
        ans.append(float(i.strip())) 
    return ans


def calculate_insurance_price(data):
    base_price = 5000  # Base price for the insurance

    data['age'] = int(data['age'])
    data['income_level'] = int(data['income_level'])
    data['previous_claims'] = int(data['previous_claims'])
    
    # Age factor
    if data['age'] < 25:
        age_factor = 1.3
    elif data['age'] < 35:
        age_factor = 1.1
    elif data['age'] < 50:
        age_factor = 1.2
    else:
        age_factor = 1.4

    # Gender factor
    gender_factor = 1.2 if data['gender'] == 'Male' else 1.0

    # City factor
    city_factors = {
        'Ahmedabad': 1.1,
        'Bangalore': 1.1,
        'Chandigarh': 1.2,
        'Chennai': 1.2,
        'Delhi': 1.3,
        'Hyderabad': 1.2,
        'Kanpur': 1.1,
        'Kolkata': 1.0,
        'Lucknow': 1.1,
        'Mumbai': 1.3,
        'Nagpur': 1.1,
        'Pune': 1.1
    }
    city_factor = city_factors.get(data['city'], 1.2)

    # Occupation factor
    occupation_factor = 0.8 if data['occupation'] == 'Active' else 1.2

    # Smoking status factor
    smoking_factor = 1.5 if data['smoking_status'] == 'Smoker' else 1.0

    # Alcohol consumption factor
    if data['alcohol_consumption'] == 'None':
        alcohol_factor = 1.0
    elif data['alcohol_consumption'] == 'Occasional':
        alcohol_factor = 1.1
    else:
        alcohol_factor = 1.3

    # Previous claims factor
    if data['previous_claims'] == 0:
        claims_factor = 1.0
    elif data['previous_claims'] == 1:
        claims_factor = 1.2
    else:
        claims_factor = 1.5

    # Income level factor
    if data['income_level'] < 300000:
        income_factor = 1.2
    elif data['income_level'] < 600000:
        income_factor = 1.0
    else:
        income_factor = 0.8

    if data['past_disease_history'] != "":
        past_disease_factor = sum(insurance(data['past_disease_history']))
    else:
        past_disease_factor = 1.0

    if data['family_disease_history'] != "":
        family_disease_factor = sum(insurance(data['past_disease_history']))
    else:
        family_disease_factor = 1.0



    # Education level factor
    education_factor = 1.1 if data['education_level'] == 'Undergraduate' else 0.9

    # Calculate final price
    final_price = (base_price * age_factor * gender_factor * city_factor * occupation_factor *
                   smoking_factor * alcohol_factor * claims_factor * income_factor *
                   past_disease_factor * family_disease_factor * education_factor)
    
    return final_price