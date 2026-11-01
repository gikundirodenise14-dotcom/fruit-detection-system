from pathlib import Path
import streamlit as st
import pickle
from ultralytics import YOLO

st.set_page_config(
    page_title="Fruit Detection System",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "data.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("data.pkl was not found.")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()
