import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.Food.data import preimage_data
load_dotenv()
from PIL import Image
import ast


class food_report_generator:

    

    def report(self,image, disease):
        def input_file(uploaded_file=None, file_path=None):
            if uploaded_file:
                # Handle file upload
                bytes_data = uploaded_file.read()
                image_parts = {
                    "mime_type": uploaded_file.mimetype,
                    "data": bytes_data
                }
                return image_parts
            elif file_path:
                # Handle file path
                with open(file_path, 'rb') as f:
                    bytes_data = f.read()
                # Here you would need to determine the MIME type, but for simplicity, it's left as 'image/jpeg'
                # In a real-world scenario, you might use the `mimetypes` library to detect the MIME type.
                image_parts = {
                    "mime_type": 'image/jpeg',
                    "data": bytes_data
                }
                return image_parts
            else:
                return None
            
        
        image = input_file(file_path=image)
        image_disease_prompt = f"""
You are an expert in nutritionist. Analyze the food items from the image and provide the nutritional value in the following format:

1. Item 1 - Nutritional value with Estimated Macro and Micro Nutrients 
2. Item 2 - Nutritional value with Estimated Macro and Micro Nutrients 
----
----
Based on the nutritional value and the disease "{disease}", tell me if this meal is good or bad for the user. Explain your reasoning.
"""
        image_prompt = f"""
You are an expert in nutritionist. Analyze the food items from the image and provide the nutritional value in the following format:

1. Item 1 - Nutritional value with Estimated Macro and Micro Nutrients 
2. Item 2 - Nutritional value with Estimated Macro and Micro Nutrients 
----
----
tell me if this meal is good or bad for the user. Explain your reasoning.
"""
        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key) 
        
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        if disease == None:
            response = model.generate_content([image, image_prompt])
        else:
            response = model.generate_content([image, image_disease_prompt])
        response = response.text


        return response