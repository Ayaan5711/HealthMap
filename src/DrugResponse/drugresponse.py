import google.generativeai as genai
import os
from dotenv import load_dotenv
import ast


class report_generator:

    def report(self,data):
        prompt = f"""You are an advanced medical AI that helps in predicting potential side effects of medications based on a patient's profile. Given the following patient data, predict 10 potential side effects that this patient might experience when taking the specified drug. Store these side effects in a Python list.

            Patient Data:{data}
            Generate the list of potential side effects with descriptions of those side effects.
            just give me dictionary without anything else
            
            """
        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key) 
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt])
        response = response.text
        response = response.replace("```python\n","")
        response = response.replace("\n```","")
        dictionary = ast.literal_eval(response)
        print(dictionary)
        return dictionary['potential_side_effects']