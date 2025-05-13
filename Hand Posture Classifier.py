import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_hand_model():
    try:
        return load_model('hand_classifier.h5')
    except FileNotFoundError:
        st.error("Error: The 'hand_classifier.h5' model file was not found. Please make sure it is in the same directory as this script.")
        return None

model = load_hand_model()

st.title("Proper vs. Improper Hand Classification")
st.write("Upload an image of a hand to classify it as 'Proper' or 'Improper'.")

file = st.file_uploader("Upload a hand image...", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert("RGB")
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0
    try:
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

if model is not None:
    if file is None:
        st.text("Please upload an image file.")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)

            if prediction is not None:
                class_names = ['improper', 'proper']
                result = class_names[np.argmax(prediction)]
                confidence = prediction[0][np.argmax(prediction)] * 100
                st.success(f"Prediction: **{result.title()}** (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
else:
    st.warning("The hand classification model could not be loaded. Please ensure 'hand_classifier.h5' is in the correct location.")
