import streamlit as st
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure GenerativeAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit UI
st.set_page_config(page_title="Medical Report Generator")
st.title("Medical Image and Medicine Image Analyzer")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image of a medical condition or a medicine", type=["jpg", "jpeg", "png"])

# Analysis type selection
analysis_type = st.radio(
    "Select analysis type:",
    ["Medical Image Analysis", "Medicine Image Analysis"],
    index=0
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Prepare the prompt based on the selected analysis type
    if analysis_type == "Medical Image Analysis":
        sample_prompt = """
        You are a medical practitioner and an expert in analyzing medical-related images. 
        You will be provided with images and you need to identify any anomalies, diseases, or health issues. 
        Generate a detailed report including findings, next steps, recommendations, and any other relevant information. 
        Include a disclaimer stating, "Consult with a Doctor before making any decisions."
        
        Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

        Analyze the image and provide a detailed report.
        """
        
    elif analysis_type == "Medicine Image Analysis":
        sample_prompt = """
        You are a medical practitioner and an expert in analyzing medicine images. 
        You will be provided with images and you need to identify the medicine type, its formula, side effects, and use cases. 
        Generate a detailed report including findings, next steps, use cases, and side effects. 
        Include a disclaimer stating, "Consult with a Doctor before making any decisions."

        Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

        Analyze the image and provide a detailed report.
        """

    # Generate response when the user clicks the "Analyze" button
    if st.button("Analyze"):
        # Read the image from the uploaded file
        img = Image.open(uploaded_file)

        # Add a loading spinner
        with st.spinner("Analyzing..."):
            # Generate response using GenerativeAI model
            # Convert the image to a suitable format for the model
            response = model.generate_content([sample_prompt, img])

        # Display the generated response
        st.subheader("Generated Medical Report:")
        st.write(response.text)

        # Download report button
        st.download_button(
            label="Download Report",
            data=BytesIO(response.text.encode('utf-8')),
            file_name=f"{analysis_type.lower().replace(' ', '_')}_report.txt",
            key="download_button"
        )
