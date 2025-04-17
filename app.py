import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
import easyocr
import numpy as np
import cv2
from PIL import Image

# UI setup
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

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

# File uploader
uploaded_file = st.file_uploader("📁 Upload a dashboard screenshot/image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Dashboard Image", use_column_width=True)

    with st.spinner("🔍 Extracting text from the image..."):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        extracted_text = " ".join([text for (_, text, _) in results])

        st.subheader("📄 Extracted Text")
        st.text_area("🧾 Text from Image", extracted_text, height=200)

    with st.spinner("🤖 Summarizing dashboard with Gemini..."):
        prompt = f"""
You are reviewing a business or analytics dashboard.

Here is the extracted text:

\"\"\"{extracted_text}\"\"\"

Summarize what the dashboard is showing — including KPIs, trends, and important metrics.
"""

        try:
            response = llm([
                SystemMessage(content="You are a professional analyst that explains dashboards."),
                HumanMessage(content=prompt)
            ])
            summary = response.content
            st.subheader("📈 Gemini Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"Gemini API error: {e}")
