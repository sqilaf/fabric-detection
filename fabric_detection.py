import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
import base64

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="AI Fabric Studio", page_icon="✨", layout="centered")

# --- 2. FUNCTION IMAGE BACKGROUND ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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
        /* Overlay gelap nipis 30% supaya background tak ganggu text */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.3); 
            z-index: -1;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
except:
    st.warning("⚠️ Background image tak jumpa. Pastikan nama fail 'background.jpg'.")

# --- 3. CSS "MEDIA STUDENT" THEME (THE AESTHETIC FIX) ---
st.markdown("""
    <style>
    /* IMPORT GOOGLE FONTS (Playfair Display untuk Tajuk, Poppins untuk text) */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins:wght@400;600&display=swap');

    /* KOTAK UTAMA (Kaca Krim) */
    .main .block-container {
        background: rgba(241, 238, 220, 0.95); /* #F1EEDC pekat sikit */
        border-radius: 25px;
        padding: 2rem 3rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        border: 2px solid #977F48;
    }

    /* 1. TAJUK (Font Baru & Hijau Gelap) */
    h1 {
        font-family: 'Playfair Display', serif; /* Font Fashion */
        color: #1A3300 !important; /* Hijau Gelap Sangat (#1A3300) */
        font-weight: 800 !important;
        text-transform: uppercase;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 5px;
        text-shadow: 1px 1px 0px #fff;
    }
    
    /* 2. SUBTITLE DALAM SHAPE (Bubble) */
    .subtitle-box {
        background-color: #1A3300; /* Background Hijau Gelap */
        color: #F1EEDC; /* Tulisan Krim */
        padding: 10px 25px;
        border-radius: 50px; /* Curve Shape */
        font-family: 'Poppins', sans-serif;
        text-align: center;
        font-size: 0.9rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    .center-box {
        text-align: center; /* Untuk letak bubble kat tengah */
    }

    /* 3. TABS (Dark Brown & Putih) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E0D8B0; /* Warna tab tak aktif */
        border-radius: 10px 10px 0px 0px;
        color: #5C4033;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3E2723 !important; /* DARK BROWN PEKAT */
        color: #FFFFFF !important; /* TULISAN PUTIH */
        border-radius: 10px 10px 0px 0px;
    }

    /* 4. FILE UPLOADER (Fix Warna Drag & Drop + Tulisan Fail) */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed #3E2723; /* Border Dark Brown */
    }
    /* Warna butang 'Browse files' */
    button[data-testid="baseButton-secondary"] {
        background-color: #3E2723; /* Dark Brown */
        color: white;
        border: none;
    }
    /* Warna text file yang diupload (BAJU COTTON.jpg) */
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF; /* Background putih untuk list file */
        border-radius: 10px;
        padding: 10px;
        color: #000000 !important; /* Paksa tulisan jadi hitam */
    }
    /* Kecilkan tulisan upload limit tu */
    small {
        color: #3E2723 !important;
        font-weight: bold;
    }

    /* BUANG HEADER/FOOTER */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER CUSTOM (DENGAN SHAPE SUBTITLE) ---
st.markdown("<h1>🧵 AI Fabric Studio</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class="center-box">
        <div class="subtitle-box">
            Kenali jenis fabrik anda: Cotton, Denim, Silk, atau Polyester ✨
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 5. LOGIC AI ---
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
    uploaded_file = st.file_uploader("Sila upload gambar kain (JPG/PNG)", type=["jpg", "png", "jpeg"], key="upload")
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
        st.image(image, caption="Analisis sedang dijalankan...", use_container_width=True)

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
    
    # --- 6. LOGIC WARNA RESULT (DYNAMIC COLOR) ---
    percentage = confidence_score * 100
    
    # Tentukan warna ikut peratusan
    if percentage >= 80:
        box_color = "#1A3300" # Hijau Gelap (Match Tajuk)
        border_color = "#32CD32"
        status_text = "SANGAT YAKIN"
    elif percentage >= 50:
        box_color = "#CC9900" # Kuning/Gold Gelap (Supaya tulisan putih nampak)
        border_color = "#FFD700"
        status_text = "AGAK YAKIN"
    else:
        box_color = "#8B0000" # Merah Gelap
        border_color = "#FF0000"
        status_text = "KURANG PASTI"

    # Custom Result Box dengan Dynamic Color
    st.markdown(f"""
        <style>
        .result-card {{
            background-color: {box_color};
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-top: 20px;
            border: 2px solid {border_color};
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
            animation: fadeIn 1s;
        }}
        @keyframes fadeIn {{
            from {{opacity: 0; transform: translateY(20px);}}
            to {{opacity: 1; transform: translateY(0);}}
        }}
        </style>
        
        <div class="result-card">
            <p style="margin:0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8;">{status_text}</p>
            <h2 style="margin: 10px 0; font-family: 'Playfair Display', serif; font-size: 2.5rem;">
                ✨ {class_name[2:].strip().upper()} ✨
            </h2>
            <div style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 15px; display: inline-block;">
                <p style="margin:0; font-weight: bold; font-family: 'Poppins', sans-serif;">
                    Score: {percentage:.2f}%
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
