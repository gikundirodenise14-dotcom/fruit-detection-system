import streamlit as st
from ultralytics import YOLO

st.title("🍎 Fruit Detection System")

@st.cache_resource
def load_model():
    return YOLO("yolo11m.pt")

try:
    model = load_model()
    st.success("YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Fruit")

    if st.button("Detect Fruits"):
        results = model.predict(uploaded_file.getvalue())

        st.image(
            results[0].plot(),
            caption="Fruit Detection Results"
        )
        
