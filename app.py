import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Load the VGG16-based model
model = load_model("./models/vgg.h5")

# Load the pre-trained VGG16 base for feature extraction
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(180, 180, 3))

# App title with a header
st.set_page_config(page_title="Chest X-Ray Pneumonia Detection", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        color: #006DAA;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
    }
    .result-box {
        text-align: center;
        padding: 20px;
        background-color: #006DAA;
        border-radius: 10px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and instructions
st.markdown("<h1 class='title'>Chest X-Ray Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a Chest X-Ray image to classify it as <strong>Normal</strong> or <strong>Pneumonia</strong>.</p>", unsafe_allow_html=True)

# Function to preprocess and extract features
def preprocess_image_and_extract_features(image):
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = image.resize((180, 180))  # Resize to match VGG16 input size
    image = np.array(image)  # Convert to numpy array
    image = preprocess_input(image)  # Apply VGG16 preprocessing
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = conv_base.predict(image)  # Extract features using VGG16
    return features

# Function to interpret predictions
def interpret_prediction(prediction, threshold=0.5):
    if prediction >= threshold:
        predicted_class = "Pneumonia"
        confidence = prediction[0][0] * 100  # Convert to percentage
    else:
        predicted_class = "Normal"
        confidence = (1 - prediction[0][0]) * 100  # Convert to percentage
    return predicted_class, confidence

# File uploader
st.markdown("<h3>Upload Chest X-Ray Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

    # Preprocess the image and extract features
    with st.spinner("Processing the image..."):
        features = preprocess_image_and_extract_features(image)

    # Make prediction using the model
    prediction = model.predict(features)

    # Interpret the prediction
    predicted_class, confidence = interpret_prediction(prediction)

    # Display the results
    st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='result-box'>
            <p style='font-size: 30px;'>{predicted_class}</p>
            <p style='font-size: 20px;'>Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add a success or warning message based on prediction
    if predicted_class == "Pneumonia":
        st.warning("Please consult a healthcare professional immediately for further assessment.")
    else:
        st.success("The chest X-ray appears normal. No further action is needed.")

else:
    st.info("Please upload a Chest X-Ray image to proceed.")

# Footer
st.markdown(
    """
    <hr style="border:1px solid #f5f5f5;">
    <p style='text-align: center; color: gray;'>Developed by Group 12 (Bhupesh & Avtar) | Powered by TensorFlow and Streamlit</p>
    """,
    unsafe_allow_html=True,
)