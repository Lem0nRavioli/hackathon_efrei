import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# load the trained model
model = load_model('best_model.h5')

# constants
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Use the same dimensions as during training

st.title("Cataract Detection")
st.write("Upload an eye image to predict if it shows signs of cataract.")

# upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # predict
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.7:
        label = "Normal"
    elif 0.4 <= prediction < 0.7:
        label = "Cataract Suspicion"
    else:
        label = "Cataract"
    
    # label = "Normal" if prediction >= 0.5 else "Cataract"
    # confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.subheader("Prediction:")
    st.write(f"**{label}** (Label: {prediction:.2f})")
    st.write("_The closer to 1 the more confident the model about your cataract (or absence of, in case of closer to 0)._")