import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
import base64

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="AI Fabric Studio", page_icon="🧵", layout="centered")

# --- 2. BACKGROUND IMAGE FUNCTION ---
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
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.4); 
            z-index: -1;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
except:
    st.warning("⚠️ Background image not found. Please ensure 'background.jpg' is in GitHub.")

# --- 3. CSS LUXURY THEME (EXTRA BOLD VARSITY) ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Graduate&family=Poppins:wght@400;600;800&display=swap');

    /* MAIN CONTAINER */
    .main .block-container {
        background: rgba(253, 232, 205, 0.95); /* BEIGE */
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        border: 3px solid #362706; /* BROWN */
    }

    /* 1. TITLE (THE "FAKE BOLD" TRICK) */
    h1 {
        font-family: 'Graduate', serif !important; 
        color: #043915 !important; /* Warna Hijau Gelap */
        font-weight: 900 !important; 
        text-transform: uppercase;
        text-align: center !important; 
        letter-spacing: 2px;
        font-size: 3.5rem !important; 
        margin-bottom: 20px;
        line-height: 1.2;
        text-shadow: none !important;
        
        /* RAHSIA GEMUKKAN TULISAN: */
        /* Kita letak outline tebal (2.5px) warna SAMA dengan tulisan */
        -webkit-text-stroke: 2.5px #043915; 
    }
    
    /* 2. SUBTITLE BUBBLE */
    .subtitle-box {
        background-color: #043915; 
        color: #F9F5F0; 
        padding: 10px 30px;
        border-radius: 50px;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        margin-bottom: 30px;
        border: 1px solid #362706;
    }
    .center-box { text-align: center; }

    /* 3. TABS DESIGN */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F9F5F0;
        border-radius: 8px 8px 0px 0px;
        color: #362706;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border: 1px solid #362706;
    }
    .stTabs [aria-selected="true"] {
        background-color: #362706 !important;
        color: #F9F5F0 !important;
    }

    /* 4. FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #F9F5F0;
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed #362706;
    }
    [data-testid="stFileUploader"] div {
        color: #000000 !important;
    }
    button[data-testid="baseButton-secondary"] {
        background-color: #362706;
        color: #F9F5F0;
        border: none;
    }
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF;
        border: 1px solid #ddd;
        color: #000000 !important;
    }

    /* CAMERA INPUT */
    div[data-testid="stCameraInput"] {
        border: 2px dashed #362706;
        background-color: #F9F5F0;
        border-radius: 15px;
    }
    button {
        background-color: #362706 !important;
        color: white !important;
    }

    /* 5. FOOTER BUBBLE */
    .footer-box {
        background-color: #FDE8CD; 
        color: #362706; 
        padding: 10px 40px;
        border-radius: 50px; 
        font-family: 'Graduate', serif; 
        text-align: center;
        font-size: 0.9rem;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        border: 2px solid #362706;
        /* Footer pun kita tebalkan sikit tapi nipis je outline dia */
        -webkit-text-stroke: 0.5px #362706;
    }

    /* HIDE STREAMLIT BRANDING */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & SUBTITLE ---
st.markdown("<h1>AI FABRIC DETECTION STUDIO</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class="center-box">
        <p style="font-size: 2rem; margin-top: -20px;">🧵</p>
        <div class="subtitle-box">
            Identify your fabric type: Cotton, Denim, Silk, or Polyester ✨
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 5. AI LOGIC ---
@st.cache_resource
def load_my_model():
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)
    with custom
