import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('cataract_mobilnetv2_finetuned_classifier.h5')

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Use the same dimensions as during training

st.title("Cataract Detection")
st.write("Upload an eye image to predict if it shows signs of cataract.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Normal" if prediction >= 0.5 else "Cataract"

    st.subheader("Prediction:")
    st.write(f"**{label}** (Confidence: {1 - prediction:.2f})")