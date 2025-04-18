import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
import easyocr
import numpy as np
import cv2
from PIL import Image
import tempfile


def analyze_dashboard_image(image_path):
    """Extracts text and numbers from the full dashboard image and prepares analysis."""
    reader = easyocr.Reader(['en'], gpu=False)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use full image for OCR
    ocr_results = reader.readtext(gray)
    extracted_text = [text[1] for text in ocr_results]
    extracted_numbers = [word for text in extracted_text for word in text.split() if word.replace(",", "").replace(".", "").isdigit()]

    description = (f"Dashboard contains charts/tables. "
                   f"Detected text: {'; '.join(extracted_text)}. "
                   f"Numbers: {', '.join(extracted_numbers)}. ")
    return description, "\n".join(extracted_text)


# Streamlit UI setup
st.set_page_config(page_title="📊 Dashboard Analyzer", layout="wide")
st.title("📊 Dashboard Image Analyzer (Gemini-powered)")

# API key input
api_key = st.text_input("🔑 Enter your Google Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()

# Configure Gemini with entered key
genai.configure(api_key=api_key)

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    top_p=0.9
)

# File uploader
uploaded_file = st.file_uploader("📁 Upload a dashboard screenshot/image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Dashboard Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    with st.spinner("🔍 Analyzing dashboard image..."):
        try:
            description, extracted_text = analyze_dashboard_image(image_path)

            st.subheader("📄 Extracted Text")
            st.text_area("💾 Text from Image", extracted_text, height=200)

            st.subheader("📊 Dashboard Summary")
            prompt = f"""You are reviewing a business or analytics dashboard.Here is the extracted text:{extracted_text}Summarize what the dashboard is showing — including KPIs, trends, and important metrics."""
            response = llm.invoke(prompt)
            st.write(response.content)

        except Exception as e:
            st.error(f"Gemini API error: {e}")
