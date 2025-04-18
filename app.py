import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
import easyocr
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
# UI setup
st.set_page_config(page_title="📊 Dashboard Analyzer", layout="wide")
st.title("📊 Dashboard Image Analyzer (Gemini-powered)")

# API key input
api_key = st.text_input("🔑 Enter your Google Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()

# Configure Gemini with entered key
os.environ["GOOGLE_API_KEY"] = api_key

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    top_p=0.9
)
st.title("Image analyzer")
st.write("Upload an image of a dashboard to analyze its contents.")
image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")

if image_path:
    image = Image.open(image_path)
    st.image(image, caption="📸 Uploaded Dashboard Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name
        reader = easyocr.Reader(['en'])
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ocr_results = reader.readtext(gray)
        extracted_text = [text[1] for text in ocr_results]
        extracted_numbers = [word for text in extracted_text for word in text.split() if word.replace(",", "").replace(".", "").isdigit()]

        # Detect active filter (Assuming a placeholder function detect_active_filter)
        #active_filter = detect_active_filter(image_path)

        description = (f"Dashboard contains charts/tables. "
                        f"Detected text: {'; '.join(extracted_text)}. "
                        f"Numbers: {', '.join(extracted_numbers)}. ")
                        #f"Active filter: {active_filter}.")

        st.write("Analyzing the image...")
        st.write(llm.invoke(f"""Your lookng into the image analysis of the dashboard based on the input summaries the dashboard and give me detailed summary of it make sure give only the information as the use will not know the whats going on the backend so use the generric so that use can only see the dashboard information
                            this is the text : {description}""").content)
        
