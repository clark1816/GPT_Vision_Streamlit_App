import streamlit as st
import os
import base64
from openai import OpenAI
import pandas as pd
import re

# Initialize OpenAI client with API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def encode_image(image_bytes):
    """Convert uploaded image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_image_analysis(image_bytes, prompt):
    """Use GPT-4 Vision to analyze the image with custom prompt."""
    base64_image = encode_image(image_bytes)

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Streamlit UI
st.title("Image Analysis with GPT-4 Vision")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Text input for custom prompt
prompt = st.text_area(
    "Enter your prompt for image analysis:",
    "Please describe what you see in this image."
)

if uploaded_file and prompt:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Add analyze button
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            # Get the image bytes
            image_bytes = uploaded_file.getvalue()
            
            # Get analysis
            analysis = get_image_analysis(image_bytes, prompt)
            
            # Display results
            st.subheader("Analysis Results:")
            st.write(analysis)
