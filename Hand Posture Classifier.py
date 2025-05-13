pip install tensorflow 

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_hand_model():
    return load_model('hand_classifier.h5')

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
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['improper', 'proper']
    result = class_names[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)] * 100
    st.success(f"Prediction: **{result.title()}** (Confidence: {confidence:.2f}%)")
