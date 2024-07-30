import google.generativeai as genai
import os
from dotenv import load_dotenv
import json


class report_generator:

    def report(self,disease, area):
        prompt = f""" {disease} on {area} area, just give me a json file containing,
                name of disease, description, most likely cause of disease, Precautions we need to take, 
                Symptoms of that disease, diet we need to follow, affect of disease, 
                  remember to always give me output in json format and 
                make sure the key name is  consistant and values are in list format"""

        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key) 
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt])
        response = response.text

        if "```json" in response:
            response = response.replace("```json","")
        if "```" in response:
            response = response.replace("```","")

        pos = response.find("\n\n\n")
        response = response[:pos]

        with open("src/data/Content.json", "w") as f:
            f.writelines(response)
        with open("src/data/Content.json","r") as file:
            data = json.load(file)

        return data