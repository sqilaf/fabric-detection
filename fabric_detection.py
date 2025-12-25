import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope

# --- 1. SETUP PAGE (WAJIB LETAK PALING ATAS) ---
st.set_page_config(page_title="AI Fashion Studio", page_icon="✨", layout="centered")

# --- 2. CSS AESTHETIC (STYLE PINTEREST / GLASSMORPHISM) ---
st.markdown("""
    <style>
    /* Masukkan Background Image (Gambar Texture Fashion) */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1558769132-cb1aea458c5e?q=80&w=2574&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Effect Kaca (Glassmorphism) untuk kotak utama */
    .main .block-container {
        background: rgba(255, 255, 255, 0.85); /* Putih tapi telus sikit */
        backdrop-filter: blur(15px); /* Effect blur belakang kaca */
        border-radius: 25px; /* Bucu bulat */
        padding: 2rem 3rem; /* Padding atas bawah dikurangkan sikit */
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); /* Bayang lembut */
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Tajuk Font Moden */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #333 !important;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Sub-tajuk */
    p {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }

    /* Style untuk Tabs biar nampak mahal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px 10px 0px 0px;
        color: #666;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.5) !important;
        color: #333 !important;
        border-bottom: 3px solid #FF4B2B !important; /* Garis merah di bawah tab aktif */
    }

    /* Hilangkan header standard Streamlit yang semak */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER CUSTOM ---
st.markdown("<h1>✨ AI Fabric Studio</h1>", unsafe_allow_html=True)
st.markdown("<p>Pilih cara input: Upload gambar atau guna kamera.</p>", unsafe_allow_html=True)

# --- 4. FUNCTION LOAD MODEL (CACHE SUPAYA LAJU) ---
@st.cache_resource
def load_my_model():
    # Fix error 'groups=1'
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

# --- 5. PILIHAN INPUT (TABS: UPLOAD VS KAMERA) ---
# Kita buat dua tab supaya kemas
tab1, tab2 = st.tabs(["📁 Upload Gambar", "📸 Guna Kamera"])

image_source = None # Variable untuk simpan gambar tak kisah dari sumber mana

with tab1:
    # Ini cara lama (Upload file)
    uploaded_file = st.file_uploader("Pilih gambar dari PC/Phone", type=["jpg", "png", "jpeg"], key="upload")
    if uploaded_file is not None:
        image_source = uploaded_file

with tab2:
    # INI CARA BARU (Webcam macam Teachable Machine)
    camera_file = st.camera_input("Senyum dan tekan butang 'Take Photo'", key="camera")
    if camera_file is not None:
        image_source = camera_file


# --- 6. PROSES GAMBAR (JIKA ADA INPUT) ---
# Kod ini akan jalan tak kisah user guna upload ATAU kamera
if image_source is not None:
    
    # Buka gambar dari sumber yang dipilih
    image = Image.open(image_source).convert("RGB")
    
    # Letak gambar tengah-tengah guna column
    st.markdown("---") # Garis pemisah
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Gambar Pilihan Anda", use_container_width=True)

    # --- PROSES UNTUK AI ---
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

    # --- TUNJUK RESULT ---
    st.success(f"✨ Ini adalah fabrik: **{class_name[2:].strip()}**")
    st.caption(f"Ketepatan AI: {confidence_score*100:.2f}%")
