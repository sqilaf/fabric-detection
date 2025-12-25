import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
import base64

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="AI Fabric Studio", page_icon="✨", layout="centered")

# --- 2. FUNCTION UNTUK BACKGROUND IMAGE ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Cuba load gambar background, kalau tak jumpa guna warna plain je
try:
    img_base64 = get_base64_of_bin_file("background.jpg")
    background_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Tambah overlay gelap sikit supaya tulisan nampak */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.2); /* Layer putih nipis */
            backdrop-filter: blur(3px); /* Blur sikit background image tu */
            z-index: -1;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
except:
    st.warning("⚠️ Sila upload gambar 'background.jpg' ke dalam GitHub anda untuk lihat background.")

# --- 3. CSS TEMA "EARTH TONE" (PALETTE AWAK) ---
st.markdown("""
    <style>
    /* KOTAK UTAMA (GLASSMORPHISM - WARNA KRIM) */
    .main .block-container {
        background: rgba(241, 238, 220, 0.9); /* Warna Krim dari Palette (#F1EEDC) */
        backdrop-filter: blur(10px); /* Blur kuat untuk kotak ni */
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 10px 30px rgba(44, 95, 45, 0.3); /* Bayang hijau gelap */
        border: 2px solid #977F48; /* Border warna Coklat Gold */
    }

    /* TAJUK (WARNA HIJAU GELAP) */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2C5F2D !important; /* Warna Hijau Gelap Palette */
        font-weight: 900 !important; /* BOLD KUAT */
        text-transform: uppercase;
        text-align: center;
        text-shadow: 2px 2px 0px #FFFFFF; /* Stroke putih supaya timbul */
        margin-bottom: 10px;
    }
    
    /* SUB-TAJUK */
    p {
        text-align: center;
        color: #588157; /* Warna Hijau Sage */
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* TABS DESIGN */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid #977F48; /* Garis Coklat */
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 10px 10px 0px 0px;
        color: #2C5F2D; /* Tulisan Hijau */
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #977F48 !important; /* Tab Aktif jadi Coklat Gold */
        color: white !important;
    }

    /* BUTTON UPLOAD & KAMERA */
    .stFileUploader, div[data-testid="stCameraInput"] {
        border: 2px dashed #2C5F2D; /* Border Hijau Putus-putus */
        border-radius: 15px;
        padding: 10px;
        background-color: rgba(255,255,255,0.7);
    }
    
    /* HILANGKAN MENU STREAMLIT */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ALERT BOX (RESULT) */
    .stAlert {
        background-color: #2C5F2D;
        color: white;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER BARU (BOLD & EMOJI) ---
st.markdown("<h1>🧵 AI FABRIC DETECTION STUDIO 🧵</h1>", unsafe_allow_html=True)
st.markdown("<p>Kenali jenis fabrik anda: Cotton, Denim, Silk, atau Polyester.</p>", unsafe_allow_html=True)

# --- 5. LOGIC AI (SAMA MACAM SEBELUM NI) ---
@st.cache_resource
def load_my_model():
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)
    with custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
        model = load_model("keras_model.h5", compile=False)
    return model

try:
    model = load_my_model()
    class_names = open("labels.txt", "r").readlines()
except Exception as e:
    st.error(f"Error loading model: {e}")

# TABS
tab1, tab2 = st.tabs(["📁 Upload Gambar", "📸 Guna Kamera"])
image_source = None

with tab1:
    uploaded_file = st.file_uploader("Upload fail gambar (JPG/PNG)", type=["jpg", "png", "jpeg"], key="upload")
    if uploaded_file is not None:
        image_source = uploaded_file

with tab2:
    camera_file = st.camera_input("Ambil gambar kain", key="camera")
    if camera_file is not None:
        image_source = camera_file

# PROSES
if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Gambar Fabrik Anda", use_container_width=True)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Custom Result Box ikut tema
    st.markdown(f"""
        <div style="background-color: #2C5F2D; padding: 20px; border-radius: 15px; text-align: center; color: white; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
            <h2 style="margin:0; color: white;">✨ {class_name[2:].strip().upper()} ✨</h2>
            <p style="margin:0; color: #F1EEDC; font-size: 1rem;">Keyakinan AI: {confidence_score*100:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
