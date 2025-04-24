import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load the trained model
model = load_model('best_model.h5')

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Use the same dimensions as during training

# Grad-CAM function
def generate_gradcam(img, model, last_conv_layer_name="block_16_project_BN"):
    """
    Generates a Grad-CAM heatmap superimposed on the input image.

    Parameters:
    - img: PIL Image object
    - model: trained Keras model
    - last_conv_layer_name: str, name of the last convolutional layer for Grad-CAM

    Returns:
    - superimposed_img: numpy array of RGB image with heatmap overlay
    """
    # Preprocess image for Grad-CAM
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize input for MobileNetV2
    
    # Create a model that maps input to (conv layer output, prediction)
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Use GradientTape to compute gradients of the target class prediction w.r.t. conv layer output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        target_class = predictions[:, 0]  # Class index 0: cataract (your case)

    # Gradient of the target class score w.r.t. feature map
    grads = tape.gradient(target_class, conv_outputs)
    
    # Take mean over height & width axes → gives importance weight for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by its importance (channel-wise attention)
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU and Normalize
    heatmap = np.maximum(heatmap.numpy(), 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # Inverse heatmap because model predicts cataract (0) as positive class
    heatmap = 1.0 - heatmap

    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert original image to BGR for OpenCV
    img_orig = np.array(img)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

    # Blend the original image with heatmap
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)
    
    return superimposed_img


# Streamlit App
st.title("Cataract Detection with Grad-CAM Visualization")
st.write("Upload an eye image to predict if it shows signs of cataract and visualize the Grad-CAM heatmap.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make a prediction
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.7:
        label = "Normal"
    elif 0.4 <= prediction < 0.7:
        label = "Cataract Suspicion"
    else:
        label = "Cataract"

    confidence = 1 - prediction

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"**{label}** (Confidence: {confidence:.2f})")
    st.write("_The closer to 1, the more confident the model is about the absence of cataract._")

    # Generate Grad-CAM image
    gradcam_img = generate_gradcam(img_resized, model)
    
    # Display Grad-CAM heatmap
    st.subheader("Grad-CAM Visualization:")
    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

    # Explanation of the heatmap
    st.markdown("""
    **Heatmap Explanation:**
    - The Grad-CAM heatmap highlights regions of the image that the model focuses on to make its prediction.
    - Areas with higher intensity (closer to red) indicate regions associated with a healthy eye (prediction closer to 1).
    - Areas with lower intensity (closer to blue) indicate regions associated with cataract (prediction closer to 0).
    """)
