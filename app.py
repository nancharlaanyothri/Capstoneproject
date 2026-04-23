import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Skin Disease Prediction",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------
# LOAD HUGGINGFACE MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    model_name = "Ateeqq/skin-disease-prediction-exp-v1"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()


# -------------------------------------------------
# THEME TOGGLE
# -------------------------------------------------
mode = st.sidebar.radio("Select Theme", ["Dark 🌙", "Light ☀️"])

if mode == "Dark 🌙":
    bg = "#0f172a"
    text = "white"
    card_bg = "#1e293b"
    sidebar_bg = "#111827"
else:
    bg = "#f8fafc"
    text = "black"
    card_bg = "white"
    sidebar_bg = "#e5e7eb"

st.markdown(f"""
<style>
/* Main App Background */
.stApp {{
    background-color: {bg};
    color: {text};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg};
}}

/* Card Styling */
.card {{
    background-color: {card_bg};
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
    color: {text};
}}

/* Buttons */
.stButton>button {{
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧬Skin Disease Prediction")


# -------------------------------------------------
# SIDEBAR PATIENT INFO
# -------------------------------------------------
st.sidebar.header("Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def predict(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label[predicted_class.item()]
    conf = round(confidence.item() * 100, 2)

    return label, conf, probs[0].numpy()

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🔍 Analyze"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        result, confidence, probabilities = predict(image)

        st.markdown("## 🧪 Diagnosis Result")

        st.markdown(f"""
        <div class="card">
            <h2 style="color:#22c55e;">{result.upper()}</h2>
            <h4>Confidence: {confidence}%</h4>
            <p>Please consult a dermatologist for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability Chart
        labels = list(model.config.id2label.values())

        prob_data = {
            labels[i]: float(probabilities[i]) * 100
            for i in range(len(labels))
        }

        st.subheader("Prediction Confidence for All Classes")
        st.bar_chart(prob_data)

    else:
        st.warning("Please upload an image first.")