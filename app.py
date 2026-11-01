from ultralytics import YOLO
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "yolo11m.pt"

@st.cache_resource
def load_model():
    return YOLO("yolo11m.pt")

model = load_model()
