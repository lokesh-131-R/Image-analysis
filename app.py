import streamlit as st
import cv2
import easyocr
from PIL import Image
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Title of the Streamlit app
st.title("📊 Dashboard Image Analyzer")

# User input for API Key
google_api_key = st.text_input("Enter your Google API Key:", type="password")

# Check if the user entered the API key
if google_api_key:
    # Set the API key as an environment variable
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Initialize the LangChain Google Generative AI
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # or "gemini-1.5-pro" if available
        temperature=0.1,
        top_p=0.9
    )

    # Upload the image
    uploaded_file = st.file_uploader("Upload your dashboard image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image locally
        image_path = "temp_uploaded_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Button to trigger the analysis
        if st.button("Analyze"):
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], gpu=False)

            # Read the image and perform OCR
            img = cv2.imread(image_path)
            ocr_results = reader.readtext(img)
            
            # Extract text from OCR results
            extracted_text = " ".join([res[1] for res in ocr_results])

            # Prepare the prompt for summarization
            prompt = f"Summarize the following dashboard content in a concise, insight-driven way:\n\n{extracted_text}"

            # Call Google Gemini (LangChain) for summarization
            gemini_response = llm.invoke(prompt).content

            # Display the Gemini summary
            st.subheader("Google Gemini Summary")
            st.write(gemini_response)
else:
    st.warning("Please enter your Google API Key to proceed.")
