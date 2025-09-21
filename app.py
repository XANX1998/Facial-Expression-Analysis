import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model("facial_emotion_model.keras")

# Emotion labels (make sure order matches train_generator.class_indices)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("😊 Facial Emotion Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess
    img = image.load_img(uploaded_file, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    pred = model.predict(img_array)
    emotion_class = np.argmax(pred)
    emotion_label = class_labels[emotion_class]

    # Show results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Emotion:** {emotion_label}")
