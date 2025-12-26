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

# --- 3. CSS LUXURY THEME (BACK TO VARSITY FONT + HEAVY STROKE) ---
st.markdown("""
    <style>
    /* IMPORT FONTS: Balik pada 'Graduate' (Ala Freshman) & Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Graduate&family=Poppins:wght@400;600;800&display=swap');

    /* MAIN CONTAINER */
    .main .block-container {
        background: rgba(253, 232, 205, 0.95); /* BEIGE */
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        border: 3px solid #362706; /* BROWN */
    }

    /* 1. TITLE (FONT GRADUATE DENGAN OUTLINE TEBAL BIAR NAMPAK BOLD) */
    h1 {
        font-family: 'Graduate', serif !important; 
        color: #043915 !important; /* Warna isi: GREEN */
        font-weight: 900 !important; /* Paling tebal */
        text-transform: uppercase;
        text-align: center;
        letter-spacing: 3px; /* Jarakkan sikit sebab outline tebal */
        font-size: 3rem !important;
        margin-bottom: 15px;
        /* RAHSIA BOLD: Tambah outline tebal warna Brown */
        -webkit-text-stroke: 2px #362706; 
        text-shadow: none;
        white-space: nowrap; 
    }
    
    /* 2. SUBTITLE BUBBLE (HIJAU) */
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

    /* 5. FOOTER BUBBLE (Guna font Graduate juga biar sekata) */
    .footer-box {
        background-color: #FDE8CD; /* BEIGE background */
        color: #362706; /* DARK BROWN text */
        padding: 10px 40px;
        border-radius: 50px; /* Curve shape */
        font-family: 'Graduate', serif; /* FONT VARSITY */
        text-align: center;
        font-size: 0.85rem;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        border: 2px solid #362706;
    }

    /* HIDE STREAMLIT BRANDING */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER & SUBTITLE ---
st.markdown("<h1>🧵 AI FABRIC DETECTION STUDIO 🧵</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class="center-box">
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
    with custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
        model = load_model("keras_model.h5", compile=False)
    return model

try:
    model = load_my_model()
    class_names = open("labels.txt", "r").readlines()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- 6. TABS ---
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Use Camera"])
image_source = None

with tab1:
    uploaded_file = st.file_uploader("Upload an image file (JPG/PNG)", type=["jpg", "png", "jpeg"], key="upload")
    if uploaded_file is not None:
        image_source = uploaded_file

with tab2:
    camera_file = st.camera_input("Take a picture of the fabric", key="camera")
    if camera_file is not None:
        image_source = camera_file

# --- 7. PROCESSING & RESULT ---
if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

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
    
    # --- DYNAMIC COLOR LOGIC ---
    percentage = confidence_score * 100
    
    if percentage >= 80:
        box_color = "#043915" # GREEN
        border_color = "#043915"
        status_text = "HIGH CONFIDENCE"
    elif percentage >= 50:
        box_color = "#F4991A" # ORANGE
        border_color = "#362706"
        status_text = "MEDIUM CONFIDENCE"
    else:
        box_color = "#CF0F0F" # RED
        border_color = "#CF0F0F"
        status_text = "LOW CONFIDENCE"

    # RESULT CARD
    st.markdown(f"""
        <style>
        .result-card {{
            background-color: {box_color};
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            color: #F9F5F0;
            margin-top: 20px;
            border: 3px solid {border_color};
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
            animation: fadeIn 1s;
        }}
        @keyframes fadeIn {{
            from {{opacity: 0; transform: translateY(20px);}}
            to {{opacity: 1; transform: translateY(0);}}
        }}
        </style>
        
        <div class="result-card">
            <p style="margin:0; font-family: 'Poppins', sans-serif; font-size: 0.9rem; letter-spacing: 2px; opacity: 0.9;">{status_text}</p>
            <h2 style="margin: 10px 0; font-family: 'Graduate', serif; font-size: 3rem; font-weight: 900; letter-spacing: 3px;">
                ✨ {class_name[2:].strip().upper()} ✨
            </h2>
            <div style="background: rgba(249, 245, 240, 0.2); padding: 5px 20px; border-radius: 15px; display: inline-block;">
                <p style="margin:0; font-weight: bold; font-family: 'Poppins', sans-serif; color: #F9F5F0;">
                    Accuracy Score: {percentage:.2f}%
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 8. FOOTER CREDIT (BEIGE BUBBLE) ---
st.markdown("""
    <div class="center-box" style="margin-top: 50px; margin-bottom: 30px;">
        <div class="footer-box">
            THIS WEBSITE CREATED BY AQILAH, AINA, AINUR, SYASYA
        </div>
    </div>
""", unsafe_allow_html=True)
