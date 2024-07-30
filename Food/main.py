import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from data import preimage_data
load_dotenv()
from PIL import Image

#genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')#gemini-1.5-flash
    response=model.generate_content([input,image[0],prompt])
    return response.text

def input_file(uploaded_file=None):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        print(image_parts)
        return image_parts
    else:
        return preimage_data
    

st.set_page_config(page_title = "Nutrition Health App")
st.header("Food Analyzer")

uploaded_file = st.file_uploader("Upload a photo of your meal", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

disease = st.text_input("Enter your disease (if any):")

input_prompt = f"""
You are an expert in nutritionist. Analyze the food items from the image and provide the nutritional value in the following format:

1. Item 1 - Nutritional value with Estimated Macro and Micro Nutrients 
2. Item 2 - Nutritional value with Estimated Macro and Micro Nutrients 
----
----

Based on the nutritional value and the disease "{disease}", tell me if this meal is good or bad for the user. Explain your reasoning.
"""

submit = st.button("Analyze Meal")

if submit:
    image_data = input_file(uploaded_file)
    with st.spinner('Wait for it...'):
        response = get_gemini_response(input_prompt, image_data, "")
    st.subheader("Meal Analysis:")
    st.write(response)
