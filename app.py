import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from sign_mapping import sign_use

model = tf.keras.models.load_model("model/road_sign_cnn.h5")
class_names = np.load("model/classes.npy", allow_pickle=True)

st.title("🚦 Road Sign Identifier")
st.write("Upload a road sign image to know its use")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((64, 64))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)

    folder_label = str(class_names[class_id])
    meaning = sign_use.get(folder_label, "Meaning not found")

    st.success(f"Detected Folder/Class: {folder_label}")
    st.info(f"Meaning: {meaning}")