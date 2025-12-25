import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope

# --- SETUP PAGE BIAR LAWA ---
st.set_page_config(page_title="AI Pengesan Kain", page_icon="👕")
st.title("👕 AI Pengesan Jenis Kain")
st.write("Upload gambar baju, dan AI akan teka jenis kainnya!")

# --- FUNCTION LOAD MODEL (CACHE SUPAYA LAJU) ---
@st.cache_resource
def load_my_model():
    # Fix error 'groups=1' macam dalam Colab tadi
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)

    with custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
        model = load_model("keras_model.h5", compile=False)
    return model

# Load model & labels
try:
    model = load_my_model()
    class_names = open("labels.txt", "r").readlines()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- FILE UPLOADER (PENGGANTI WEBCAM/CODING MANUAL) ---
file = st.file_uploader("Sila upload gambar baju (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is not None:
    # Tunjuk gambar yang user upload
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # --- PROSES GAMBAR (SAMA MACAM COLAB) ---
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- PREDICT ---
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # --- TUNJUK RESULT CANTIK-CANTIK ---
    st.success(f"🎉 Ini adalah: **{class_name[2:].strip()}**")
    st.info(f"Tahap Keyakinan AI: {confidence_score*100:.2f}%")
